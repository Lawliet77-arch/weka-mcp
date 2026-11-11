package dev.wekamcp.analysis;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.Preprocessing;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.DatasetEntry;
import dev.wekamcp.registry.SessionRegistry.ModelEntry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.RankedOutputSearch;
import weka.attributeSelection.Ranker;
import weka.associations.Apriori;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class AnalysisService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;

    public AnalysisService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks) {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
    }

    public ObjectNode profileDataset(String datasetId) {
        DatasetEntry entry = registry.dataset(datasetId);
        Instances data = entry.data();
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("instances", data.numInstances());
        result.put("attributes", data.numAttributes());
        result.put("class_index", data.classIndex() < 0 ? -1 : data.classIndex() + 1);
        Set<String> rows = new HashSet<>();
        long missingCells = 0;
        for (Instance instance : data) {
            rows.add(instance.toString());
            for (int index = 0; index < data.numAttributes(); index++) {
                if (instance.isMissing(index)) {
                    missingCells++;
                }
            }
        }
        result.put("duplicate_rows", data.numInstances() - rows.size());
        result.put("missing_cells", missingCells);
        ToolSupport.putNumber(result, "missing_percent", percentage(missingCells,
                (double) data.numInstances() * data.numAttributes()));
        ArrayNode attributes = result.putArray("attributes");
        for (int index = 0; index < data.numAttributes(); index++) {
            Attribute attribute = data.attribute(index);
            ObjectNode item = attributes.addObject();
            item.put("index", index + 1);
            item.put("name", attribute.name());
            item.put("type", attributeType(attribute));
            Set<String> unique = new HashSet<>();
            Map<String, Integer> nominalCounts = new HashMap<>();
            List<Double> numeric = new ArrayList<>();
            long missing = 0;
            for (Instance instance : data) {
                if (instance.isMissing(index)) {
                    missing++;
                    continue;
                }
                unique.add(instance.toString(index));
                if (attribute.isNumeric()) {
                    numeric.add(instance.value(index));
                }
                if (attribute.isNominal()) {
                    nominalCounts.merge(instance.stringValue(index), 1, Integer::sum);
                }
            }
            item.put("missing", missing);
            ToolSupport.putNumber(item, "missing_percent", percentage(missing, data.numInstances()));
            item.put("unique_count", unique.size());
            item.put("constant", unique.size() <= 1);
            if (attribute.isNominal()) {
                item.set("nominal_counts", ToolSupport.JSON.valueToTree(nominalCounts));
            }
            if (!numeric.isEmpty()) {
                numeric.sort(Double::compare);
                double mean = numeric.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
                double variance = numeric.stream().mapToDouble(value -> (value - mean) * (value - mean)).sum()
                        / Math.max(1, numeric.size() - 1);
                ToolSupport.putNumber(item, "mean", mean);
                ToolSupport.putNumber(item, "std_dev", Math.sqrt(variance));
                ToolSupport.putNumber(item, "min", numeric.get(0));
                ToolSupport.putNumber(item, "q1", quantile(numeric, 0.25));
                ToolSupport.putNumber(item, "median", quantile(numeric, 0.5));
                ToolSupport.putNumber(item, "q3", quantile(numeric, 0.75));
                ToolSupport.putNumber(item, "max", numeric.get(numeric.size() - 1));
            }
        }
        if (data.classIndex() >= 0 && data.classAttribute().isNominal()) {
            Map<String, Integer> counts = new HashMap<>();
            for (Instance instance : data) {
                counts.merge(instance.classAttribute().value((int) instance.classValue()), 1, Integer::sum);
            }
            result.set("class_distribution", ToolSupport.JSON.valueToTree(counts));
            int min = counts.values().stream().min(Integer::compare).orElse(0);
            int max = counts.values().stream().max(Integer::compare).orElse(0);
            ToolSupport.putNumber(result, "class_imbalance_ratio", min == 0 ? Double.POSITIVE_INFINITY : (double) max / min);
        }
        addCorrelations(result, data);
        return result;
    }

    public ObjectNode learningCurve(String datasetId, String classifierClass, String[] options, int classIndex,
            String fractions, long seed, boolean autoPreprocess) throws Exception {
        DatasetEntry entry = registry.dataset(datasetId);
        Instances data = WekaTools.classificationData(entry, classIndex);
        WekaTools.DataSplit outer = WekaTools.splitData(data, 80.0, seed);
        String[] parts = fractions.split(",");
        ArrayNode curves = ToolSupport.array();
        for (String part : parts) {
            double fraction = Double.parseDouble(part.trim());
            if (fraction <= 0 || fraction > 1) {
                throw new IllegalArgumentException("fractions must be in (0, 1]");
            }
            WekaTools.DataSplit subset = WekaTools.splitData(outer.training(), fraction * 100.0, seed);
            Instances training = subset.training();
            Classifier base = (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER, classifierClass, options);
            Classifier classifier = Preprocessing.wrap(base,
                    autoPreprocess ? Preprocessing.automatic(training, classifierClass) : List.of(), whitelist);
            ObjectNode point = ToolSupport.object();
            point.put("fraction", fraction);
            point.put("training_instances", training.numInstances());
            point.put("test_instances", outer.test().numInstances());
            tasks.call(config.trainTimeoutSeconds(), () -> {
                classifier.buildClassifier(training);
                Evaluation evaluation = new Evaluation(training);
                evaluation.evaluateModel(classifier, outer.test());
                WekaTools.addEvaluationMetrics(point, evaluation, outer.test());
                return null;
            });
            curves.add(point);
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("classifier_class", classifierClass);
        result.set("options", ToolSupport.JSON.valueToTree(options));
        result.put("seed", seed);
        result.put("auto_preprocess", autoPreprocess);
        result.set("curves", curves);
        return result;
    }

    public ObjectNode modelDiagnostics(String modelId, String datasetId, int bins, int maxErrors) throws Exception {
        if (bins < 1 || bins > 100 || maxErrors < 0 || maxErrors > 1000) {
            throw new IllegalArgumentException("bins must be 1..100 and max_errors must be 0..1000");
        }
        ModelEntry model = registry.model(modelId);
        DatasetEntry entry = registry.dataset(datasetId);
        Instances input = new Instances(entry.data());
        input.setClassIndex(model.trainingHeader().classIndex());
        if (!model.trainingHeader().equalHeaders(input)) {
            throw new IllegalArgumentException("Diagnostics dataset header is incompatible with the training header");
        }
        ObjectNode result = ToolSupport.object();
        result.put("model_id", model.id());
        result.put("dataset_id", entry.id());
        if (model.experimentId() != null) {
            result.put("experiment_id", model.experimentId());
        }
        Evaluation evaluation = new Evaluation(input);
        synchronized (model) {
            evaluation.evaluateModel(model.classifier(), input);
        }
        WekaTools.addEvaluationMetrics(result, evaluation, input);
        if (input.classAttribute().isNominal()) {
            addNominalDiagnostics(result, model, input, bins, maxErrors);
        } else {
            addRegressionDiagnostics(result, model, input, bins, maxErrors);
        }
        return result;
    }

    public ObjectNode thresholdAnalysis(Map<String, Object> arguments) throws Exception {
        ModelEntry model = registry.model(ToolSupport.requiredString(arguments, "model_id"));
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        Instances input = new Instances(entry.data());
        input.setClassIndex(model.trainingHeader().classIndex());
        if (!model.trainingHeader().equalHeaders(input)) {
            throw new IllegalArgumentException("Threshold dataset header is incompatible with the training header");
        }
        if (input.numInstances() > config.maxPredictionRows()) {
            throw new IllegalArgumentException("Threshold analysis rows exceed maxPredictionRows=" + config.maxPredictionRows());
        }
        if (input.classAttribute().numValues() != 2) {
            throw new IllegalArgumentException("Threshold analysis requires a binary nominal class");
        }
        double falsePositiveCost = ToolSupport.optionalDouble(arguments, "cost_fp", 1.0);
        double falseNegativeCost = ToolSupport.optionalDouble(arguments, "cost_fn", 1.0);
        if (falsePositiveCost < 0 || falseNegativeCost < 0 || falsePositiveCost + falseNegativeCost == 0) {
            throw new IllegalArgumentException("cost_fp and cost_fn must be nonnegative and not both zero");
        }
        String positiveClass = ToolSupport.optionalString(arguments, "positive_class", input.classAttribute().value(1));
        int positiveIndex = input.classAttribute().indexOfValue(positiveClass);
        if (positiveIndex < 0) {
            throw new IllegalArgumentException("Unknown positive_class: " + positiveClass);
        }
        int bins = ToolSupport.optionalInt(arguments, "bins", 50);
        int steps = ToolSupport.optionalInt(arguments, "steps", 21);
        if (bins < 1 || bins > 100 || steps < 2 || steps > 201) {
            throw new IllegalArgumentException("bins must be 1..100 and steps must be 2..201");
        }
        java.util.TreeSet<Double> thresholds = new java.util.TreeSet<>();
        String requestedThresholds = ToolSupport.optionalString(arguments, "thresholds", null);
        if (requestedThresholds != null && !requestedThresholds.isBlank()) {
            for (String part : requestedThresholds.split(",")) {
                double threshold = Double.parseDouble(part.trim());
                if (!Double.isFinite(threshold) || threshold < 0 || threshold > 1) {
                    throw new IllegalArgumentException("thresholds must be finite probabilities in [0, 1]");
                }
                thresholds.add(threshold);
            }
        } else {
            for (int index = 0; index < steps; index++) {
                thresholds.add(index / (double) (steps - 1));
            }
        }
        if (thresholds.isEmpty() || thresholds.size() > 200) {
            throw new IllegalArgumentException("thresholds must contain between 1 and 200 probabilities");
        }
        record BinaryPrediction(double positive, int actual, double margin, boolean correct) {
        }
        List<BinaryPrediction> predictions = new ArrayList<>();
        int[] skipped = new int[2];
        tasks.call(config.trainTimeoutSeconds(), () -> {
            synchronized (model) {
                for (Instance instance : input) {
                    if (instance.classIsMissing()) {
                        skipped[0]++;
                        continue;
                    }
                    double[] distribution = model.classifier().distributionForInstance(instance);
                    double positive = distribution[positiveIndex];
                    int actual = (int) instance.classValue();
                    double other = Double.NEGATIVE_INFINITY;
                    for (int index = 0; index < distribution.length; index++) {
                        if (index != actual) {
                            other = Math.max(other, distribution[index]);
                        }
                    }
                    if (!Double.isFinite(positive) || !Double.isFinite(other)) {
                        skipped[1]++;
                        continue;
                    }
                    predictions.add(new BinaryPrediction(positive, actual, distribution[actual] - other,
                            (positive >= 0.5 ? positiveIndex : 1 - positiveIndex) == actual));
                }
            }
            return null;
        });
        ObjectNode result = ToolSupport.object();
        result.put("model_id", model.id());
        result.put("dataset_id", entry.id());
        result.put("positive_class", positiveClass);
        result.put("instances", input.numInstances());
        result.put("scored_instances", predictions.size());
        result.put("skipped_missing_class", skipped[0]);
        result.put("unclassified", skipped[1]);
        ToolSupport.putNumber(result, "cost_fp", falsePositiveCost);
        ToolSupport.putNumber(result, "cost_fn", falseNegativeCost);
        ObjectNode best = null;
        ArrayNode curve = result.putArray("thresholds");
        for (double threshold : thresholds) {
            int tp = 0;
            int fp = 0;
            int fn = 0;
            int tn = 0;
            for (BinaryPrediction prediction : predictions) {
                if (prediction.positive >= threshold) {
                    if (prediction.actual == positiveIndex) {
                        tp++;
                    } else {
                        fp++;
                    }
                } else if (prediction.actual == positiveIndex) {
                    fn++;
                } else {
                    tn++;
                }
            }
            ObjectNode item = curve.addObject();
            ToolSupport.putNumber(item, "threshold", threshold);
            item.put("tp", tp);
            item.put("fp", fp);
            item.put("fn", fn);
            item.put("tn", tn);
            ToolSupport.putNumber(item, "accuracy_percent", percentage(tp + tn, predictions.size()));
            ToolSupport.putNumber(item, "precision", ratio(tp, tp + fp));
            ToolSupport.putNumber(item, "recall", ratio(tp, tp + fn));
            ToolSupport.putNumber(item, "false_positive_rate", ratio(fp, fp + tn));
            ToolSupport.putNumber(item, "f1", ratio(2 * tp, 2 * tp + fp + fn));
            double cost = fp * falsePositiveCost + fn * falseNegativeCost;
            ToolSupport.putNumber(item, "cost", cost);
            if (best == null || cost < best.get("cost").doubleValue()
                    || cost == best.get("cost").doubleValue()
                    && Math.abs(threshold - 0.5) < Math.abs(best.get("threshold").doubleValue() - 0.5)) {
                best = item;
            }
        }
        result.set("best", best);
        result.put("best_threshold", best.get("threshold").doubleValue());
        result.put("minimum_cost", best.get("cost").doubleValue());
        ArrayNode histogram = result.putArray("margin_histogram");
        for (int bin = 0; bin < bins; bin++) {
            double lower = -1.0 + 2.0 * bin / bins;
            double upper = -1.0 + 2.0 * (bin + 1) / bins;
            int correct = 0;
            int incorrect = 0;
            for (BinaryPrediction prediction : predictions) {
                if (prediction.margin >= lower && (prediction.margin < upper || bin == bins - 1 && prediction.margin <= upper)) {
                    if (prediction.correct) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
            }
            ObjectNode item = histogram.addObject();
            ToolSupport.putNumber(item, "lower", lower);
            ToolSupport.putNumber(item, "upper", upper);
            item.put("correct", correct);
            item.put("incorrect", incorrect);
        }
        return result;
    }

    private static double ratio(double numerator, double denominator) {
        return denominator == 0 ? Double.NaN : numerator / denominator;
    }

    public ObjectNode selectAttributes(String datasetId, int classIndex, String method, int maximum) throws Exception {
        DatasetEntry entry = registry.dataset(datasetId);
        Instances data = WekaTools.classificationData(entry, classIndex);
        if (maximum < 1 || maximum >= data.numAttributes()) {
            throw new IllegalArgumentException("max_attributes must be between 1 and attributes - 1");
        }
        AttributeSelection selection = new AttributeSelection();
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("method", method);
        result.put("max_attributes", maximum);
        if ("info_gain".equals(method)) {
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            ranker.setNumToSelect(maximum);
            selection.setEvaluator(evaluator);
            selection.setSearch(ranker);
            selection.SelectAttributes(data);
            ArrayNode ranking = result.putArray("ranking");
            int rankCount = 0;
            for (double[] ranked : ((RankedOutputSearch) ranker).rankedAttributes()) {
                if (rankCount++ >= maximum) {
                    break;
                }
                ObjectNode item = ranking.addObject();
                item.put("index", (int) ranked[0] + 1);
                item.put("name", data.attribute((int) ranked[0]).name());
                ToolSupport.putNumber(item, "info_gain", ranked[1]);
            }
        } else if ("cfs".equals(method)) {
            CfsSubsetEval evaluator = new CfsSubsetEval();
            GreedyStepwise search = new GreedyStepwise();
            search.setNumToSelect(maximum);
            selection.setEvaluator(evaluator);
            selection.setSearch(search);
            selection.SelectAttributes(data);
        } else {
            weka.attributeSelection.ASEvaluation evaluator = switch (method) {
                case "gain_ratio" -> new weka.attributeSelection.GainRatioAttributeEval();
                case "symmetrical_uncert" -> new weka.attributeSelection.SymmetricalUncertAttributeEval();
                case "one_r" -> new weka.attributeSelection.OneRAttributeEval();
                case "relief_f" -> new weka.attributeSelection.ReliefFAttributeEval();
                default -> throw new IllegalArgumentException(
                        "method must be info_gain, cfs, gain_ratio, symmetrical_uncert, one_r or relief_f");
            };
            Ranker ranker = new Ranker();
            ranker.setNumToSelect(maximum);
            selection.setEvaluator(evaluator);
            selection.setSearch(ranker);
            selection.SelectAttributes(data);
            ArrayNode ranking = result.putArray("ranking");
            int rankCount = 0;
            for (double[] ranked : ((RankedOutputSearch) ranker).rankedAttributes()) {
                if (rankCount++ >= maximum) {
                    break;
                }
                ObjectNode item = ranking.addObject();
                item.put("index", (int) ranked[0] + 1);
                item.put("name", data.attribute((int) ranked[0]).name());
                ToolSupport.putNumber(item, "score", ranked[1]);
            }
        }
        int[] selected = selection.selectedAttributes();
        ArrayNode attributes = result.putArray("selected_attributes");
        for (int index : selected) {
            if (index == data.classIndex()) {
                continue;
            }
            ObjectNode item = attributes.addObject();
            item.put("index", index + 1);
            item.put("name", data.attribute(index).name());
        }
        return result;
    }

    public ObjectNode associate(String datasetId, String algorithm, double minSupport, double minConfidence,
            int maxRules, boolean autoPrepare, String[] options) throws Exception {
        DatasetEntry entry = registry.dataset(datasetId);
        if (!Set.of("apriori", "fpgrowth").contains(algorithm)) {
            throw new IllegalArgumentException("algorithm must be apriori or fpgrowth");
        }
        if (minSupport <= 0 || minSupport > 1 || minConfidence < 0 || minConfidence > 1) {
            throw new IllegalArgumentException("min_support must be in (0, 1] and min_confidence must be in [0, 1]");
        }
        if (maxRules < 1 || maxRules > 1000) {
            throw new IllegalArgumentException("max_rules must be 1..1000");
        }
        String className = "fpgrowth".equals(algorithm) ? "weka.associations.FPGrowth" : "weka.associations.Apriori";
        Instances associationData = autoPrepare ? prepareAssociationData(entry.data(), algorithm) : entry.data();
        weka.associations.Associator associator = (weka.associations.Associator) whitelist.create(
                Whitelist.Kind.ASSOCIATOR, className, options);
        weka.associations.AssociationRules associationRules = tasks.call(config.trainTimeoutSeconds(), () -> {
            if (associator instanceof Apriori apriori) {
                if (options.length == 0) {
                    apriori.setLowerBoundMinSupport(minSupport);
                    apriori.setMinMetric(minConfidence);
                    apriori.setNumRules(maxRules);
                }
            } else {
                weka.associations.FPGrowth fpGrowth = (weka.associations.FPGrowth) associator;
                if (options.length == 0) {
                    fpGrowth.setLowerBoundMinSupport(minSupport);
                    fpGrowth.setMinMetric(minConfidence);
                    fpGrowth.setNumRulesToFind(maxRules);
                }
            }
            associator.buildAssociations(associationData);
            return ((weka.associations.AssociationRulesProducer) associator).getAssociationRules();
        });
        double actualSupport;
        double actualConfidence;
        int ruleLimit;
        if (associator instanceof Apriori apriori) {
            actualSupport = apriori.getLowerBoundMinSupport();
            actualConfidence = apriori.getMinMetric();
            ruleLimit = apriori.getNumRules();
        } else {
            weka.associations.FPGrowth fpGrowth = (weka.associations.FPGrowth) associator;
            actualSupport = fpGrowth.getLowerBoundMinSupport();
            actualConfidence = fpGrowth.getMinMetric();
            ruleLimit = fpGrowth.getNumRulesToFind();
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("algorithm", algorithm);
        result.put("auto_prepare", autoPrepare);
        result.put("attributes_after_preparation", associationData.numAttributes());
        ToolSupport.putNumber(result, "min_support", actualSupport);
        ToolSupport.putNumber(result, "min_confidence", actualConfidence);
        result.set("options", ToolSupport.JSON.valueToTree(((weka.core.OptionHandler) associator).getOptions()));
        ArrayNode rules = result.putArray("rules");
        for (weka.associations.AssociationRule rule : associationRules.getRules()) {
            if (rules.size() >= ruleLimit) {
                break;
            }
            ObjectNode item = rules.addObject();
            item.put("premise", rule.getPremise().toString());
            item.put("consequence", rule.getConsequence().toString());
            ToolSupport.putNumber(item, "support", (double) rule.getTotalSupport() / rule.getTotalTransactions());
            ToolSupport.putNumber(item, "confidence", rule.getNamedMetricValue("Confidence"));
            ToolSupport.putNumber(item, "lift", rule.getNamedMetricValue("Lift"));
        }
        result.put("rule_count", rules.size());
        result.put("truncated", associationRules.getRules().size() > rules.size());
        return result;
    }

    private Instances prepareAssociationData(Instances input, String algorithm) throws Exception {
        Instances data = new Instances(input);
        boolean numeric = false;
        for (int index = 0; index < data.numAttributes(); index++) {
            Attribute attribute = data.attribute(index);
            numeric |= attribute.isNumeric();
            if (attribute.isString() || attribute.type() == Attribute.RELATIONAL) {
                throw new IllegalArgumentException("Association analysis requires numeric or nominal attributes");
            }
        }
        if (numeric) {
            weka.filters.Filter discretize = (weka.filters.Filter) whitelist.create(Whitelist.Kind.FILTER,
                    "weka.filters.unsupervised.attribute.Discretize", new String[]{"-B", "10"});
            discretize.setInputFormat(data);
            data = weka.filters.Filter.useFilter(data, discretize);
        }
        if ("fpgrowth".equals(algorithm)) {
            long expandedAttributes = 0;
            boolean multiValued = false;
            for (int index = 0; index < data.numAttributes(); index++) {
                Attribute attribute = data.attribute(index);
                multiValued |= attribute.isNominal() && attribute.numValues() > 2;
                expandedAttributes += attribute.isNominal() && attribute.numValues() > 2 ? attribute.numValues() : 1;
            }
            if (expandedAttributes > config.maxAttributes()) {
                throw new IllegalArgumentException("Nominal binary expansion exceeds maxAttributes=" + config.maxAttributes());
            }
            if (multiValued) {
                weka.filters.unsupervised.attribute.NominalToBinary binary =
                        (weka.filters.unsupervised.attribute.NominalToBinary) whitelist.create(Whitelist.Kind.FILTER,
                                "weka.filters.unsupervised.attribute.NominalToBinary", new String[0]);
                binary.setBinaryAttributesNominal(true);
                binary.setInputFormat(data);
                data = weka.filters.Filter.useFilter(data, binary);
            }
        }
        return data;
    }

    public ObjectNode explainModel(String modelId) throws Exception {
        ModelEntry model = registry.model(modelId);
        Classifier classifier = model.classifier();
        if (classifier instanceof FilteredClassifier filtered) {
            classifier = filtered.getClassifier();
        }
        ObjectNode result = ToolSupport.object();
        result.put("model_id", model.id());
        result.put("classifier_class", model.classifierClass());
        if (model.experimentId() != null) {
            result.put("experiment_id", model.experimentId());
        }
        if (classifier instanceof J48 tree) {
            result.put("supported", true);
            result.put("format", "j48_text_and_dot");
            result.put("tree", tree.toString());
            result.put("dot", tree.graph());
            ToolSupport.putNumber(result, "tree_size", tree.measureTreeSize());
            ToolSupport.putNumber(result, "num_leaves", tree.measureNumLeaves());
            ToolSupport.putNumber(result, "num_rules", tree.measureNumRules());
        } else {
            result.put("supported", false);
            result.put("reason", "This classifier does not expose a J48-style tree or DOT representation");
        }
        return result;
    }

    private static void addCorrelations(ObjectNode result, Instances data) {
        List<Integer> numeric = new ArrayList<>();
        for (int index = 0; index < data.numAttributes(); index++) {
            if (data.attribute(index).isNumeric() && index != data.classIndex()) {
                numeric.add(index);
            }
        }
        if (numeric.size() > 50) {
            return;
        }
        List<Correlation> correlations = new ArrayList<>();
        for (int left = 0; left < numeric.size(); left++) {
            for (int right = left + 1; right < numeric.size(); right++) {
                double value = correlation(data, numeric.get(left), numeric.get(right));
                if (Double.isFinite(value)) {
                    correlations.add(new Correlation(
                            data.attribute(numeric.get(left)).name(), data.attribute(numeric.get(right)).name(), value));
                }
            }
        }
        correlations.sort((left, right) -> Double.compare(Math.abs(right.value), Math.abs(left.value)));
        ArrayNode output = result.putArray("correlations");
        correlations.stream().limit(50).forEach(correlation -> {
            ObjectNode item = output.addObject();
            item.put("left", correlation.left);
            item.put("right", correlation.right);
            ToolSupport.putNumber(item, "pearson", correlation.value);
        });
    }

    private static double correlation(Instances data, int left, int right) {
        double leftMean = 0;
        double rightMean = 0;
        int count = 0;
        for (Instance instance : data) {
            if (!instance.isMissing(left) && !instance.isMissing(right)) {
                leftMean += instance.value(left);
                rightMean += instance.value(right);
                count++;
            }
        }
        if (count < 2) {
            return Double.NaN;
        }
        leftMean /= count;
        rightMean /= count;
        double covariance = 0;
        double leftVariance = 0;
        double rightVariance = 0;
        for (Instance instance : data) {
            if (!instance.isMissing(left) && !instance.isMissing(right)) {
                double leftDelta = instance.value(left) - leftMean;
                double rightDelta = instance.value(right) - rightMean;
                covariance += leftDelta * rightDelta;
                leftVariance += leftDelta * leftDelta;
                rightVariance += rightDelta * rightDelta;
            }
        }
        return leftVariance == 0 || rightVariance == 0 ? Double.NaN : covariance / Math.sqrt(leftVariance * rightVariance);
    }

    private static void addNominalDiagnostics(ObjectNode result, ModelEntry model, Instances data, int bins, int maxErrors)
            throws Exception {
        List<ScoreRow> rows = new ArrayList<>();
        synchronized (model) {
            for (int index = 0; index < data.numInstances(); index++) {
                Instance instance = data.instance(index);
                double[] scores = model.classifier().distributionForInstance(instance);
                rows.add(new ScoreRow(index, (int) instance.classValue(), Utils.maxIndex(scores), scores,
                        scores[Utils.maxIndex(scores)]));
            }
        }
        ArrayNode curves = result.putArray("curves");
        for (int classIndex = 0; classIndex < data.classAttribute().numValues(); classIndex++) {
            ObjectNode curve = curves.addObject();
            curve.put("class_label", data.classAttribute().value(classIndex));
            curve.set("precision_recall", curve(rows, classIndex, false));
            curve.set("roc", curve(rows, classIndex, true));
        }
        ArrayNode calibration = result.putArray("calibration");
        for (int bin = 0; bin < bins; bin++) {
            final int currentBin = bin;
            List<ScoreRow> members = rows.stream()
                    .filter(row -> currentBin == Math.min(bins - 1, (int) (row.confidence * bins))).toList();
            if (!members.isEmpty()) {
                ObjectNode item = calibration.addObject();
                ToolSupport.putNumber(item, "lower", (double) bin / bins);
                ToolSupport.putNumber(item, "upper", (double) (bin + 1) / bins);
                item.put("count", members.size());
                ToolSupport.putNumber(item, "mean_confidence",
                        members.stream().mapToDouble(row -> row.confidence).average().orElse(Double.NaN));
                ToolSupport.putNumber(item, "accuracy", members.stream()
                        .mapToDouble(row -> row.actual == row.predicted ? 1.0 : 0.0).average().orElse(Double.NaN));
            }
        }
        ArrayNode errors = result.putArray("errors");
        rows.stream().filter(row -> row.actual != row.predicted)
                .sorted(Comparator.comparingDouble((ScoreRow row) -> row.confidence).reversed())
                .limit(maxErrors)
                .forEach(row -> {
                    ObjectNode item = errors.addObject();
                    item.put("index", row.index);
                    item.put("actual", data.classAttribute().value(row.actual));
                    item.put("predicted", data.classAttribute().value(row.predicted));
                    ToolSupport.putNumber(item, "confidence", row.confidence);
                    item.set("distribution", ToolSupport.JSON.valueToTree(row.scores));
                });
    }

    private static ArrayNode curve(List<ScoreRow> rows, int classIndex, boolean roc) {
        List<ScoreRow> sorted = new ArrayList<>(rows);
        sorted.sort((left, right) -> Double.compare(right.scores[classIndex], left.scores[classIndex]));
        int positives = (int) rows.stream().filter(row -> row.actual == classIndex).count();
        int negatives = rows.size() - positives;
        ArrayNode points = ToolSupport.array();
        double truePositive = 0;
        double falsePositive = 0;
        int limit = Math.min(100, sorted.size());
        int step = Math.max(1, sorted.size() / limit);
        for (int index = 0; index < sorted.size(); index++) {
            ScoreRow row = sorted.get(index);
            if (row.actual == classIndex) {
                truePositive++;
            } else {
                falsePositive++;
            }
            if (index % step == 0 || index == sorted.size() - 1) {
                ObjectNode point = points.addObject();
                double tpr = positives == 0 ? 0 : truePositive / positives;
                double fpr = negatives == 0 ? 0 : falsePositive / negatives;
                ToolSupport.putNumber(point, roc ? "fpr" : "recall", roc ? fpr : tpr);
                ToolSupport.putNumber(point, roc ? "tpr" : "precision", roc ? tpr
                        : truePositive + falsePositive == 0 ? 1.0 : truePositive / (truePositive + falsePositive));
                ToolSupport.putNumber(point, "threshold", row.scores[classIndex]);
            }
        }
        return points;
    }

    private static void addRegressionDiagnostics(ObjectNode result, ModelEntry model, Instances data,
            int bins, int maxErrors) throws Exception {
        List<RegressionRow> errors = new ArrayList<>();
        synchronized (model) {
            for (int index = 0; index < data.numInstances(); index++) {
                Instance instance = data.instance(index);
                double predicted = model.classifier().classifyInstance(instance);
                double actual = instance.classValue();
                errors.add(new RegressionRow(index, actual, predicted, actual - predicted));
            }
        }
        double mean = errors.stream().mapToDouble(row -> row.residual).average().orElse(Double.NaN);
        double mae = errors.stream().mapToDouble(row -> Math.abs(row.residual)).average().orElse(Double.NaN);
        double rmse = Math.sqrt(errors.stream().mapToDouble(row -> row.residual * row.residual).average().orElse(0));
        ToolSupport.putNumber(result, "mean_residual", mean);
        ToolSupport.putNumber(result, "mae", mae);
        ToolSupport.putNumber(result, "rmse", rmse);
        ToolSupport.putNumber(result, "max_absolute_error",
                errors.stream().mapToDouble(row -> Math.abs(row.residual)).max().orElse(Double.NaN));
        ArrayNode topErrors = result.putArray("largest_errors");
        errors.stream().sorted((left, right) -> Double.compare(Math.abs(right.residual), Math.abs(left.residual)))
                .limit(maxErrors).forEach(row -> {
                    ObjectNode item = topErrors.addObject();
                    item.put("index", row.index);
                    ToolSupport.putNumber(item, "actual", row.actual);
                    ToolSupport.putNumber(item, "predicted", row.predicted);
                    ToolSupport.putNumber(item, "residual", row.residual);
                });
        addResidualHistogram(result, errors, bins);
    }

    private static void addResidualHistogram(ObjectNode result, List<RegressionRow> rows, int bins) {
        double min = rows.stream().mapToDouble(row -> row.residual).min().orElse(0);
        double max = rows.stream().mapToDouble(row -> row.residual).max().orElse(0);
        double width = max == min ? 1 : (max - min) / bins;
        ArrayNode histogram = result.putArray("residual_histogram");
        for (int bin = 0; bin < bins; bin++) {
            double lower = min + bin * width;
            double upper = bin == bins - 1 ? max : min + (bin + 1) * width;
            final int currentBin = bin;
            long count = rows.stream().filter(row -> currentBin == bins - 1
                    ? row.residual >= lower && row.residual <= upper
                    : row.residual >= lower && row.residual < upper).count();
            if (count > 0) {
                ObjectNode item = histogram.addObject();
                ToolSupport.putNumber(item, "lower", lower);
                ToolSupport.putNumber(item, "upper", upper);
                item.put("count", count);
            }
        }
    }

    private static double quantile(List<Double> sorted, double probability) {
        return sorted.get((int) Math.round((sorted.size() - 1) * probability));
    }

    private static double percentage(double value, double total) {
        return total == 0 ? 0 : value * 100.0 / total;
    }

    private static String attributeType(Attribute attribute) {
        return switch (attribute.type()) {
            case Attribute.NUMERIC -> "numeric";
            case Attribute.NOMINAL -> "nominal";
            case Attribute.STRING -> "string";
            case Attribute.DATE -> "date";
            case Attribute.RELATIONAL -> "relational";
            default -> "unknown";
        };
    }

    private record Correlation(String left, String right, double value) {
    }

    private record ScoreRow(int index, int actual, int predicted, double[] scores, double confidence) {
    }

    private record RegressionRow(int index, double actual, double predicted, double residual) {
    }
}
