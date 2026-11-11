package dev.wekamcp.evaluation;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.Preprocessing;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.DatasetEntry;
import dev.wekamcp.registry.SessionRegistry.ModelEntry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Statistics;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class ModelComparisonService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;

    public ModelComparisonService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks) {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
    }

    public ObjectNode compare(Map<String, Object> arguments) throws Exception {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        Object value = arguments.get("model_ids");
        if (!(value instanceof List<?> ids) || ids.size() < 2 || ids.size() > 8) {
            throw new IllegalArgumentException("model_ids must contain between 2 and 8 UUIDs");
        }
        List<ModelEntry> models = new ArrayList<>();
        for (Object id : ids) {
            if (!(id instanceof String modelId)) {
                throw new IllegalArgumentException("model_ids must contain only strings");
            }
            ModelEntry model = registry.model(modelId);
            if (!model.retrainable()) {
                throw new IllegalArgumentException("Imported models cannot be retrained for repeated cross-validation");
            }
            models.add(model);
        }
        Instances data = WekaTools.classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
        String metric = ToolSupport.optionalString(arguments, "metric",
                data.classAttribute().isNominal() ? "accuracy" : "rmse");
        if (!Set.of("accuracy", "weighted_f1", "rmse").contains(metric)) {
            throw new IllegalArgumentException("metric must be accuracy, weighted_f1 or rmse");
        }
        if (data.classAttribute().isNominal() != !"rmse".equals(metric)) {
            throw new IllegalArgumentException("accuracy and weighted_f1 require a nominal class; rmse requires a numeric class");
        }
        int folds = ToolSupport.optionalInt(arguments, "folds", WekaTools.automaticFoldCount(data));
        int runs = ToolSupport.optionalInt(arguments, "runs", 3);
        long seed = ToolSupport.optionalLong(arguments, "seed", 1L);
        double confidence = ToolSupport.optionalDouble(arguments, "confidence", 0.95);
        long budget = ToolSupport.optionalLong(arguments, "budgetSeconds", config.autoMlBudgetSeconds());
        if (folds < 2 || folds > data.numInstances() || folds > 100) {
            throw new IllegalArgumentException("folds must be between 2 and min(100, the number of instances)");
        }
        if (runs < 1 || runs > 20) {
            throw new IllegalArgumentException("runs must be between 1 and 20");
        }
        if (confidence < 0.5 || confidence >= 1.0) {
            throw new IllegalArgumentException("confidence must be in [0.5, 1)");
        }
        if (budget < 1 || budget > 86_400) {
            throw new IllegalArgumentException("budgetSeconds must be between 1 and 86400");
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(budget);
        double[][] scores = new double[models.size()][runs * folds];
        int completed = 0;
        boolean partial = false;
        outer:
        for (int run = 0; run < runs; run++) {
            Instances randomized = new Instances(data);
            randomized.randomize(new Random(seed + run));
            if (data.classAttribute().isNominal()) {
                randomized.stratify(folds);
            }
            for (int fold = 0; fold < folds; fold++) {
                Instances training = randomized.trainCV(folds, fold);
                Instances test = randomized.testCV(folds, fold);
                double[] foldScores = new double[models.size()];
                for (int modelIndex = 0; modelIndex < models.size(); modelIndex++) {
                    if (remainingSeconds(deadline) <= 0) {
                        partial = true;
                        break outer;
                    }
                    ModelEntry model = models.get(modelIndex);
                    Classifier classifier = Preprocessing.wrap(
                            (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER,
                                    model.classifierClass(), model.options()),
                            model.preprocessing(), whitelist);
                    try {
                        foldScores[modelIndex] = call(deadline, () -> {
                            classifier.buildClassifier(training);
                            Evaluation evaluation = new Evaluation(training);
                            evaluation.evaluateModel(classifier, test);
                            return score(evaluation, metric);
                        });
                    } catch (IllegalStateException error) {
                        if (remainingSeconds(deadline) > 0) {
                            throw error;
                        }
                        partial = true;
                        break outer;
                    }
                }
                for (int modelIndex = 0; modelIndex < models.size(); modelIndex++) {
                    scores[modelIndex][completed] = foldScores[modelIndex];
                }
                completed++;
            }
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("protocol", data.classAttribute().isNominal()
                ? "repeated_stratified_cross_validation" : "repeated_cross_validation");
        result.put("shared_folds", true);
        result.put("fold_safe_preprocessing", true);
        result.put("folds", folds);
        result.put("runs", runs);
        result.put("seed", seed);
        result.put("metric", metric);
        ToolSupport.putNumber(result, "confidence", confidence);
        result.put("fold_comparisons", completed);
        result.put("partial", partial);
        ArrayNode modelNodes = result.putArray("models");
        for (int modelIndex = 0; modelIndex < models.size(); modelIndex++) {
            ModelEntry model = models.get(modelIndex);
            double[] modelScores = java.util.Arrays.copyOf(scores[modelIndex], completed);
            ObjectNode item = modelNodes.addObject();
            item.put("model_id", model.id());
            item.put("classifier_class", model.classifierClass());
            ToolSupport.putNumber(item, "mean", mean(modelScores));
            ToolSupport.putNumber(item, "stddev", stddev(modelScores));
            ToolSupport.putNumber(item, "standard_error", stddev(modelScores) / Math.sqrt(completed));
            item.set("fold_scores", ToolSupport.JSON.valueToTree(modelScores));
        }
        double alpha = 1.0 - confidence;
        ArrayNode comparisons = result.putArray("comparisons");
        for (int modelIndex = 1; modelIndex < models.size(); modelIndex++) {
            double[] baseline = java.util.Arrays.copyOf(scores[0], completed);
            double[] other = java.util.Arrays.copyOf(scores[modelIndex], completed);
            double[] differences = new double[completed];
            for (int index = 0; index < completed; index++) {
                differences[index] = "rmse".equals(metric)
                        ? baseline[index] - other[index] : other[index] - baseline[index];
            }
            ObjectNode comparison = comparisons.addObject();
            comparison.put("baseline_model_id", models.get(0).id());
            comparison.put("other_model_id", models.get(modelIndex).id());
            comparison.put("method", "corrected_resampled_paired_t_test");
            comparison.put("degrees_of_freedom", Math.max(0, completed - 1));
            ToolSupport.putNumber(comparison, "mean_difference", mean(differences));
            ToolSupport.putNumber(comparison, "difference_stddev", stddev(differences));
            double variance = variance(differences);
            double correctedVariance = variance * (completed == 0 ? Double.NaN : 1.0 / completed + 1.0 / (folds - 1));
            double standardError = Math.sqrt(correctedVariance);
            double difference = mean(differences);
            double t = standardError == 0 ? 0 : difference / standardError;
            double probability = completed < 2 ? Double.NaN
                    : standardError == 0 ? difference == 0 ? 1.0 : 0.0 : twoSidedTProbability(t, completed - 1);
            ToolSupport.putNumber(comparison, "standard_error", standardError);
            ToolSupport.putNumber(comparison, "t_statistic", t);
            ToolSupport.putNumber(comparison, "p_value", probability);
            ToolSupport.putNumber(comparison, "alpha", alpha);
            if (completed > 1 && standardError > 0) {
                double critical = studentCritical(alpha, completed - 1);
                ToolSupport.putNumber(comparison, "confidence_interval_low", difference - critical * standardError);
                ToolSupport.putNumber(comparison, "confidence_interval_high", difference + critical * standardError);
            }
            comparison.put("winner", completed < 2 ? "insufficient_data"
                    : probability < alpha ? difference > 0 ? "other" : "baseline" : "tie");
            comparison.put("positive_difference_means", "other_is_better");
        }
        return result;
    }

    public ArrayNode repeat(Instances data, Classifier classifier, String mode, int folds,
            double trainPercent, long seed, int repeats) throws Exception {
        ArrayNode results = ToolSupport.array();
        for (int repeat = 0; repeat < repeats; repeat++) {
            long repeatSeed = seed + repeat;
            Evaluation evaluation;
            if ("cross_validation".equals(mode)) {
                evaluation = new Evaluation(data);
                evaluation.crossValidateModel(classifier, data, folds, new Random(repeatSeed));
            } else {
                WekaTools.DataSplit split = WekaTools.splitData(data, trainPercent, repeatSeed);
                Classifier copy = (Classifier) new weka.core.SerializedObject(classifier).getObject();
                copy.buildClassifier(split.training());
                evaluation = new Evaluation(split.training());
                evaluation.evaluateModel(copy, split.test());
            }
            ObjectNode item = results.addObject();
            item.put("repeat", repeat + 1);
            item.put("seed", repeatSeed);
            WekaTools.addEvaluationMetrics(item, evaluation, data);
        }
        return results;
    }

    public static void addRepeatedMetrics(ObjectNode result, ArrayNode repeated, Instances data) {
        result.put("aggregation", "mean_of_repeats");
        result.put("repeats", repeated.size());
        result.set("repeat_metrics", repeated);
        if (data.classAttribute().isNominal()) {
            average(result, repeated, "accuracy_percent");
            average(result, repeated, "kappa");
            sum(result, repeated, "correct");
            sum(result, repeated, "incorrect");
            sum(result, repeated, "unclassified");
            ArrayNode matrix = result.putArray("confusion_matrix");
            int classes = data.classAttribute().numValues();
            for (int row = 0; row < classes; row++) {
                ArrayNode values = matrix.addArray();
                for (int column = 0; column < classes; column++) {
                    double sum = 0;
                    for (JsonNode item : repeated) {
                        sum += item.get("confusion_matrix").get(row).get(column).doubleValue();
                    }
                    values.add(sum);
                }
            }
            ArrayNode perClass = result.putArray("per_class");
            for (int index = 0; index < classes; index++) {
                ObjectNode item = perClass.addObject();
                item.put("class_label", data.classAttribute().value(index));
                sum(item, repeated, "support", index);
                average(item, repeated, "precision", index);
                average(item, repeated, "recall", index);
                average(item, repeated, "f1", index);
                average(item, repeated, "auc", index);
                average(item, repeated, "false_positive_rate", index);
            }
            result.set("class_labels", ToolSupport.JSON.valueToTree(ToolSupport.classLabels(data)));
        } else {
            average(result, repeated, "mae");
            average(result, repeated, "rmse");
            average(result, repeated, "correlation_coefficient");
            average(result, repeated, "rae_percent");
            average(result, repeated, "rrse_percent");
        }
    }

    private static void average(ObjectNode result, ArrayNode repeated, String name) {
        double total = 0;
        int count = 0;
        for (JsonNode item : repeated) {
            if (item.hasNonNull(name)) {
                total += item.get(name).doubleValue();
                count++;
            }
        }
        ToolSupport.putNumber(result, name, count == 0 ? Double.NaN : total / count);
    }

    private static void average(ObjectNode result, ArrayNode repeated, String name, int index) {
        double total = 0;
        int count = 0;
        for (JsonNode item : repeated) {
            JsonNode value = item.get("per_class").get(index).get(name);
            if (value != null && !value.isNull()) {
                total += value.doubleValue();
                count++;
            }
        }
        ToolSupport.putNumber(result, name, count == 0 ? Double.NaN : total / count);
    }

    private static void sum(ObjectNode result, ArrayNode repeated, String name) {
        double total = 0;
        for (JsonNode item : repeated) {
            if (item.hasNonNull(name)) {
                total += item.get(name).doubleValue();
            }
        }
        result.put(name, total);
    }

    private static void sum(ObjectNode result, ArrayNode repeated, String name, int index) {
        double total = 0;
        for (JsonNode item : repeated) {
            total += item.get("per_class").get(index).get(name).doubleValue();
        }
        result.put(name, total);
    }

    private double call(long deadline, java.util.concurrent.Callable<Double> task) throws Exception {
        long remaining = remainingSeconds(deadline);
        if (remaining <= 0) {
            throw new IllegalStateException("Model comparison budget exhausted");
        }
        double value = tasks.call(Math.min(config.trainTimeoutSeconds(), remaining), task);
        if (!Double.isFinite(value)) {
            throw new IllegalStateException("Evaluation produced a non-finite score");
        }
        return value;
    }

    private static long remainingSeconds(long deadline) {
        return Math.max(0, (deadline - System.nanoTime() + 999_999_999L) / 1_000_000_000L);
    }

    private static double score(Evaluation evaluation, String metric) throws Exception {
        return switch (metric) {
            case "rmse" -> evaluation.rootMeanSquaredError();
            case "weighted_f1" -> evaluation.weightedFMeasure();
            default -> evaluation.pctCorrect();
        };
    }

    private static double mean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return values.length == 0 ? Double.NaN : sum / values.length;
    }

    private static double variance(double[] values) {
        if (values.length < 2) {
            return Double.NaN;
        }
        double mean = mean(values);
        double sum = 0;
        for (double value : values) {
            sum += (value - mean) * (value - mean);
        }
        return sum / (values.length - 1);
    }

    private static double stddev(double[] values) {
        return Math.sqrt(variance(values));
    }

    static double twoSidedTProbability(double t, int degreesOfFreedom) {
        if (degreesOfFreedom < 1 || !Double.isFinite(t)) {
            return Double.NaN;
        }
        return regularizedBeta(degreesOfFreedom / 2.0, 0.5,
                degreesOfFreedom / (degreesOfFreedom + t * t));
    }

    private static double regularizedBeta(double a, double b, double x) {
        if (x <= 0) {
            return 0;
        }
        if (x >= 1) {
            return 1;
        }
        double front = Math.exp(Statistics.lnGamma(a + b) - Statistics.lnGamma(a) - Statistics.lnGamma(b)
                + a * Math.log(x) + b * Math.log1p(-x));
        if (x < (a + 1) / (a + b + 2)) {
            return front * betaContinuedFraction(a, b, x) / a;
        }
        return 1 - front * betaContinuedFraction(b, a, 1 - x) / b;
    }

    private static double betaContinuedFraction(double a, double b, double x) {
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.abs(d) < 1e-300) {
            d = 1e-300;
        }
        d = 1 / d;
        double h = d;
        for (int iteration = 1; iteration <= 300; iteration++) {
            int twice = iteration * 2;
            double factor = iteration * (b - iteration) * x / ((qam + twice) * (a + twice));
            d = 1 + factor * d;
            if (Math.abs(d) < 1e-300) {
                d = 1e-300;
            }
            c = 1 + factor / c;
            if (Math.abs(c) < 1e-300) {
                c = 1e-300;
            }
            d = 1 / d;
            h *= d * c;
            factor = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice));
            d = 1 + factor * d;
            if (Math.abs(d) < 1e-300) {
                d = 1e-300;
            }
            c = 1 + factor / c;
            if (Math.abs(c) < 1e-300) {
                c = 1e-300;
            }
            d = 1 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1) < 3e-14) {
                return h;
            }
        }
        throw new IllegalStateException("Incomplete Beta failed to converge");
    }

    private static double studentCritical(double alpha, int degreesOfFreedom) {
        double low = 0;
        double high = 1;
        while (twoSidedTProbability(high, degreesOfFreedom) > alpha && high < 1_000_000) {
            high *= 2;
        }
        for (int iteration = 0; iteration < 70; iteration++) {
            double middle = (low + high) / 2;
            if (twoSidedTProbability(middle, degreesOfFreedom) > alpha) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return (low + high) / 2;
    }
}
