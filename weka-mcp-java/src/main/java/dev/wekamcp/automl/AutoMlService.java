package dev.wekamcp.automl;

import dev.wekamcp.automl.SearchSpace.SearchCandidate;
import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.Preprocessing;
import dev.wekamcp.engine.Preprocessing.FilterSpec;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.experiment.ExperimentRecord;
import dev.wekamcp.experiment.ExperimentStore;
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
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

public final class AutoMlService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;
    private final ExperimentStore experiments;
    private final SearchSpace searchSpace;

    public AutoMlService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks,
            ExperimentStore experiments) throws IOException {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
        this.experiments = experiments;
        this.searchSpace = SearchSpace.load(config.searchSpacePath());
    }

    public ObjectNode autoClassify(Map<String, Object> arguments) throws Exception {
        long started = System.nanoTime();
        long deadline = started + java.util.concurrent.TimeUnit.SECONDS.toNanos(
                longValue(arguments, "budgetSeconds", config.autoMlBudgetSeconds()));
        DatasetEntry entry = registry.dataset(required(arguments, "dataset_id"));
        String requestedClass = stringValue(arguments, "classifier_class", "auto");
        boolean automatic = "auto".equals(requestedClass);
        if (automatic && arguments.containsKey("options")) {
            throw new IllegalArgumentException("options requires an explicit classifier_class");
        }
        boolean autoPreprocess = booleanValue(arguments, "auto_preprocess", true);
        boolean runEvaluation = booleanValue(arguments, "evaluate", true);
        Instances data = WekaTools.classificationData(entry, intValue(arguments, "class_index", 0));
        if (runEvaluation && data.numInstances() < 2) {
            throw new IllegalArgumentException("Automatic evaluation requires at least two instances");
        }
        String protocol = arguments.containsKey("protocol")
                ? stringValue(arguments, "protocol", "holdout")
                : "cross_validation".equals(stringValue(arguments, "mode", "")) ? "legacy_cv" : "holdout";
        if (!java.util.Set.of("holdout", "nested_cv", "legacy_cv").contains(protocol)) {
            throw new IllegalArgumentException("protocol must be holdout, nested_cv or legacy_cv");
        }
        String search = stringValue(arguments, "search", "defaults");
        String optimize = stringValue(arguments, "optimize", data.classAttribute().isNominal() ? "accuracy" : "rmse");
        if (!java.util.Set.of("accuracy", "weighted_f1", "rmse").contains(optimize)) {
            throw new IllegalArgumentException("optimize must be accuracy, weighted_f1 or rmse");
        }
        int maxCandidates = intValue(arguments, "maxCandidates", 12);
        long seed = longValue(arguments, "seed", 1L);
        if (maxCandidates < 1 || maxCandidates > 1000) {
            throw new IllegalArgumentException("maxCandidates must be between 1 and 1000");
        }

        Object custom = arguments.get("candidates");
        if (custom != null && (!automatic || !runEvaluation)) {
            throw new IllegalArgumentException("candidates requires classifier_class=auto and evaluate=true");
        }
        if (custom != null && !(custom instanceof List<?>)) {
            throw new IllegalArgumentException("candidates must be an array of candidate objects");
        }
        List<SearchCandidate> candidates;
        if (automatic && runEvaluation) {
            candidates = custom == null
                    ? searchSpace.candidates(WekaTools.defaultCandidates(data), search, maxCandidates, seed)
                    : SearchSpace.customCandidates((List<?>) custom, maxCandidates, seed, search,
                            booleanValue(arguments, "include_baseline", true),
                            config.maxOptionsLength(), config.maxOptionTokenLength());
        } else {
            String className = automatic ? "weka.classifiers.trees.RandomForest" : requestedClass;
            candidates = List.of(new SearchCandidate(className, ToolSupport.options(
                    arguments, config.maxOptionsLength(), config.maxOptionTokenLength())));
        }

        if ("halving".equals(search) && (!automatic || !runEvaluation)) {
            throw new IllegalArgumentException("halving search requires classifier_class=auto and evaluate=true");
        }
        NestedHalvingResult nested = "halving".equals(search) && "nested_cv".equals(protocol)
                ? evaluateNestedHalving(data, candidates, arguments, autoPreprocess, optimize, seed, deadline)
                : null;
        SearchOutcome outcome = nested != null ? nested.outcome : "halving".equals(search)
                ? evaluateSuccessiveHalving(data, candidates, arguments, protocol, autoPreprocess, optimize, seed,
                        automatic, deadline)
                : evaluateCandidates(data, candidates, arguments, protocol, autoPreprocess, optimize, seed,
                        automatic, deadline);
        CandidateEvaluation selected = outcome.selected;
        ArrayNode candidateResults = outcome.candidates;
        String lastError = nested == null ? outcome.lastError : nested.lastError;
        boolean partial = nested == null ? outcome.partial : nested.partial;
        int completed = nested == null ? outcome.completed : nested.completedEvaluations;
        if (runEvaluation && selected == null) {
            throw new IllegalStateException("No candidate classifier could be evaluated"
                    + (lastError == null ? "" : ": " + lastError));
        }
        SearchCandidate selectedCandidate = selected == null ? candidates.get(0) : selected.candidate;
        ObjectNode testMetrics = nested == null ? null : nested.testMetrics;
        double testScore = Double.NaN;
        List<FilterSpec> preprocessing = autoPreprocess
                ? Preprocessing.automatic(data, selectedCandidate.classifierClass())
                : List.of();
        Classifier finalClassifier = newClassifier(selectedCandidate, preprocessing);
        boolean finalModelTrained = false;
        try {
            testMetrics = selected == null ? null : nested != null ? testMetrics : "legacy_cv".equals(protocol)
                    ? selected.validationMetrics
                    : evaluateTest(data, selectedCandidate, arguments, protocol, autoPreprocess, seed, deadline);
            testScore = selected == null ? Double.NaN : score(testMetrics, data, optimize);
            call(deadline, () -> {
                finalClassifier.buildClassifier(data);
                return null;
            });
            finalModelTrained = true;
        } catch (IllegalStateException e) {
            if (remainingSeconds(deadline) > 0) {
                throw e;
            }
            partial = true;
        }
        ExperimentRecord saved = null;
        ModelEntry model = null;
        if (finalModelTrained) {
            ExperimentRecord record = new ExperimentRecord();
            record.classifierClass = selectedCandidate.classifierClass();
            record.options = selectedCandidate.options();
            record.preprocessing = preprocessing;
            record.classIndex = data.classIndex();
            record.instances = data.numInstances();
            record.attributes = data.numAttributes();
            record.seed = seed;
            record.protocol = protocol;
            record.optimize = optimize;
            record.selectionMetric = optimize;
            record.validationScore = selected == null ? Double.NaN : selected.validationScore;
            record.testScore = testScore;
            record.validationMetrics = selected == null ? null : selected.validationMetrics;
            record.testMetrics = testMetrics;
            record.candidates = candidateResults;
            ObjectNode searchMetadata = ToolSupport.object();
            searchMetadata.put("search", search);
            searchMetadata.put("maxCandidates", maxCandidates);
            searchMetadata.put("budgetSeconds", longValue(arguments, "budgetSeconds", config.autoMlBudgetSeconds()));
            searchMetadata.put("partial", partial);
            if (outcome.stages != null) {
                searchMetadata.set("stages", outcome.stages);
            }
            if (nested != null) {
                searchMetadata.set("outer_folds", nested.outerFolds);
                searchMetadata.put("nested_cv_complete", !nested.partial);
                searchMetadata.put("selection_source", nested.selectionSource);
            }
            record.search = searchMetadata;
            saved = experiments.create(record, data, finalClassifier, data);
            model = registry.registerModel(selectedCandidate.classifierClass(), selectedCandidate.options(),
                    finalClassifier, data, preprocessing, saved.id);
        }

        ObjectNode result = ToolSupport.object();
        if (finalModelTrained) {
            result.put("model_id", model.id());
            result.put("experiment_id", saved.id);
        } else {
            result.putNull("model_id");
            result.putNull("experiment_id");
        }
        result.put("final_model_status", finalModelTrained ? "TRAINED" : "NOT_TRAINED");
        result.put("classifier_class", selectedCandidate.classifierClass());
        result.put("auto_selected", automatic && runEvaluation);
        result.put("selection_metric", optimize);
        ToolSupport.putNumber(result, "selection_score", selected == null ? Double.NaN : selected.validationScore);
        ToolSupport.putNumber(result, "validation_score", selected == null ? Double.NaN : selected.validationScore);
        ToolSupport.putNumber(result, "test_score", testScore);
        result.put("protocol", protocol);
        result.put("search", search);
        result.put("completed_candidates", completed);
        result.put("candidate_count", candidates.size());
        result.put("partial", partial);
        if (outcome.stages != null) {
            result.set("halving_stages", outcome.stages);
        }
        if (nested != null) {
            result.set("nested_outer_folds", nested.outerFolds);
            result.put("nested_cv_complete", !nested.partial);
            result.put("selection_source", nested.selectionSource);
        }
        result.put("elapsed_ms", (System.nanoTime() - started) / 1_000_000);
        result.put("auto_preprocess", autoPreprocess);
        result.set("preprocessing", ToolSupport.JSON.valueToTree(preprocessing));
        result.set("validation", selected == null ? null : selected.validationMetrics);
        result.set("evaluation", testMetrics);
        result.set("test", testMetrics);
        result.set("candidates", candidateResults);
        if (model != null) {
            result.set("model", WekaTools.modelSummary(model));
        } else {
            result.putNull("model");
        }
        return result;
    }

    private SearchOutcome evaluateCandidates(Instances data, List<SearchCandidate> candidates,
            Map<String, Object> arguments, String protocol, boolean autoPreprocess, String optimize, long seed,
            boolean automatic, long deadline) throws Exception {
        CandidateEvaluation selected = null;
        ArrayNode candidateResults = ToolSupport.array();
        String lastError = null;
        boolean partial = false;
        int completed = 0;
        for (SearchCandidate candidate : candidates) {
            if (remainingSeconds(deadline) <= 0) {
                partial = true;
                lastError = "AutoML budget exhausted";
                break;
            }
            ObjectNode candidateResult = candidateResults.addObject();
            candidateResult.put("classifier_class", candidate.classifierClass());
            candidateResult.set("options", ToolSupport.JSON.valueToTree(candidate.options()));
            try {
                CandidateEvaluation evaluation = evaluateCandidate(data, candidate, arguments, protocol, autoPreprocess,
                        optimize, seed, deadline);
                completed++;
                candidateResult.put("status", "OK");
                ToolSupport.putNumber(candidateResult, "validation_score", evaluation.validationScore);
                candidateResult.set("validation", evaluation.validationMetrics);
                if (better(evaluation, selected, optimize)) {
                    selected = evaluation;
                }
            } catch (Exception e) {
                if (!automatic) {
                    throw e;
                }
                lastError = errorMessage(e);
                candidateResult.put("status", "ERROR");
                candidateResult.put("error", lastError);
            }
        }
        return new SearchOutcome(selected, candidateResults, lastError, partial, completed, null);
    }

    private SearchOutcome evaluateSuccessiveHalving(Instances data, List<SearchCandidate> candidates,
            Map<String, Object> arguments, String protocol, boolean autoPreprocess, String optimize, long seed,
            boolean automatic, long deadline) throws Exception {
        Instances resourcePool = "holdout".equals(protocol) ? split(arguments, data, seed).training() : data;
        if (resourcePool.numInstances() < 4) {
            throw new IllegalArgumentException("halving search requires at least four training instances");
        }
        Map<String, Object> stageArguments = new LinkedHashMap<>(arguments);
        if ("nested_cv".equals(protocol)) {
            stageArguments.put("folds", intValue(arguments, "inner_folds", WekaTools.automaticFoldCount(resourcePool)));
        }
        ArrayNode candidateResults = ToolSupport.array();
        ArrayNode stages = ToolSupport.array();
        List<CandidateEvaluation> active = null;
        CandidateEvaluation selected = null;
        String lastError = null;
        boolean partial = false;
        int completed = 0;
        double[] fractions = {0.25, 0.5, 1.0};
        for (int stageIndex = 0; stageIndex < fractions.length; stageIndex++) {
            if (remainingSeconds(deadline) <= 0) {
                partial = true;
                lastError = "AutoML budget exhausted";
                break;
            }
            double fraction = fractions[stageIndex];
            Instances subset = resourceSubset(resourcePool, fraction, seed + stageIndex);
            List<SearchCandidate> stageCandidates = active == null
                    ? candidates : active.stream().map(CandidateEvaluation::candidate).toList();
            List<CandidateEvaluation> successful = new ArrayList<>();
            for (SearchCandidate candidate : stageCandidates) {
                if (remainingSeconds(deadline) <= 0) {
                    partial = true;
                    lastError = "AutoML budget exhausted";
                    break;
                }
                ObjectNode candidateResult = candidateResults.addObject();
                candidateResult.put("classifier_class", candidate.classifierClass());
                candidateResult.set("options", ToolSupport.JSON.valueToTree(candidate.options()));
                candidateResult.put("stage", stageIndex + 1);
                ToolSupport.putNumber(candidateResult, "resource_fraction", fraction);
                candidateResult.put("resource_instances", subset.numInstances());
                try {
                    CandidateEvaluation evaluation = evaluateLegacyValidation(subset, candidate, stageArguments,
                            autoPreprocess, optimize, seed, deadline);
                    if (!Double.isFinite(evaluation.validationScore)) {
                        throw new IllegalStateException("Candidate produced a non-finite validation score");
                    }
                    successful.add(evaluation);
                    completed++;
                    candidateResult.put("status", "OK");
                    ToolSupport.putNumber(candidateResult, "validation_score", evaluation.validationScore);
                    candidateResult.set("validation", evaluation.validationMetrics);
                } catch (Exception e) {
                    if (!automatic) {
                        throw e;
                    }
                    lastError = errorMessage(e);
                    candidateResult.put("status", "ERROR");
                    candidateResult.put("error", lastError);
                }
            }
            if (successful.isEmpty()) {
                partial = true;
                break;
            }
            successful.sort((left, right) -> better(left, right, optimize) ? -1
                    : better(right, left, optimize) ? 1 : 0);
            active = successful;
            selected = successful.get(0);
            int survivors = stageIndex == fractions.length - 1 ? successful.size() : (successful.size() + 1) / 2;
            List<CandidateEvaluation> retained = successful.subList(0, survivors);
            ObjectNode stage = stages.addObject();
            stage.put("stage", stageIndex + 1);
            ToolSupport.putNumber(stage, "resource_fraction", fraction);
            stage.put("resource_instances", subset.numInstances());
            stage.put("completed_candidates", successful.size());
            stage.set("retained_candidates", candidateIdentities(retained));
            stage.set("eliminated_candidates", candidateIdentities(successful.subList(survivors, successful.size())));
            active = new ArrayList<>(retained);
            if (stageIndex < fractions.length - 1 && successful.size() < stageCandidates.size()) {
                partial = true;
            }
        }
        return new SearchOutcome(selected, candidateResults, lastError, partial, completed, stages);
    }

    private static Instances resourceSubset(Instances data, double fraction, long seed) {
        if (fraction >= 1.0) {
            return data;
        }
        Instances randomized = new Instances(data);
        randomized.randomize(new Random(seed));
        int folds = (int) Math.round(1.0 / fraction);
        if (data.classAttribute().isNominal()) {
            randomized.stratify(folds);
        }
        Instances subset = randomized.testCV(folds, 0);
        subset.setClassIndex(data.classIndex());
        return subset;
    }

    private static ArrayNode candidateIdentities(List<CandidateEvaluation> candidates) {
        ArrayNode result = ToolSupport.array();
        for (CandidateEvaluation evaluation : candidates) {
            ObjectNode identity = result.addObject();
            identity.put("classifier_class", evaluation.candidate.classifierClass());
            identity.set("options", ToolSupport.JSON.valueToTree(evaluation.candidate.options()));
            ToolSupport.putNumber(identity, "validation_score", evaluation.validationScore);
        }
        return result;
    }

    private static boolean better(CandidateEvaluation candidate, CandidateEvaluation selected, String optimize) {
        if (candidate == null) {
            return false;
        }
        if (selected == null) {
            return true;
        }
        return "rmse".equals(optimize)
                ? candidate.validationScore < selected.validationScore
                : candidate.validationScore > selected.validationScore;
    }

    private static String errorMessage(Exception error) {
        return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
    }

    private NestedHalvingResult evaluateNestedHalving(Instances data, List<SearchCandidate> candidates,
            Map<String, Object> arguments, boolean autoPreprocess, String optimize, long seed, long deadline)
            throws Exception {
        int outerFolds = intValue(arguments, "outer_folds", Math.min(5, WekaTools.automaticFoldCount(data)));
        int innerFolds = intValue(arguments, "inner_folds", WekaTools.automaticFoldCount(data));
        if (outerFolds < 2 || outerFolds > data.numInstances() || innerFolds < 2) {
            throw new IllegalArgumentException("Invalid outer_folds or inner_folds");
        }
        Instances randomized = new Instances(data);
        randomized.randomize(new Random(seed));
        if (data.classAttribute().isNominal()) {
            randomized.stratify(outerFolds);
        }
        Evaluation aggregate = new Evaluation(data);
        ArrayNode outerResults = ToolSupport.array();
        ArrayNode outerCandidates = ToolSupport.array();
        boolean partial = false;
        String lastError = null;
        int completedEvaluations = 0;
        int completedFolds = 0;
        CandidateEvaluation fallback = null;
        for (int fold = 0; fold < outerFolds; fold++) {
            if (remainingSeconds(deadline) <= 0) {
                partial = true;
                lastError = "AutoML budget exhausted";
                break;
            }
            Instances training = randomized.trainCV(outerFolds, fold);
            Instances test = randomized.testCV(outerFolds, fold);
            if (innerFolds > training.numInstances()) {
                throw new IllegalArgumentException("inner_folds exceeds an outer training fold size");
            }
            SearchOutcome foldOutcome = evaluateSuccessiveHalving(training, candidates, arguments, "nested_cv",
                    autoPreprocess, optimize, seed + fold, true, deadline);
            completedEvaluations += foldOutcome.completed;
            partial |= foldOutcome.partial;
            if (foldOutcome.lastError != null) {
                lastError = foldOutcome.lastError;
            }
            if (foldOutcome.selected == null) {
                partial = true;
                continue;
            }
            CandidateEvaluation winner = foldOutcome.selected;
            fallback = winner;
            for (JsonNode candidate : foldOutcome.candidates) {
                ((ObjectNode) candidate).put("scope", "outer_training_fold");
                ((ObjectNode) candidate).put("outer_fold", fold + 1);
                outerCandidates.add(candidate);
            }
            ObjectNode foldResult = outerResults.addObject();
            foldResult.put("outer_fold", fold + 1);
            foldResult.put("classifier_class", winner.candidate.classifierClass());
            foldResult.set("options", ToolSupport.JSON.valueToTree(winner.candidate.options()));
            ToolSupport.putNumber(foldResult, "validation_score", winner.validationScore);
            foldResult.put("completed_stages", foldOutcome.stages.size());
            try {
                Classifier classifier = newClassifier(winner.candidate, autoPreprocess, training);
                call(deadline, () -> {
                    classifier.buildClassifier(training);
                    return null;
                });
                call(deadline, () -> {
                    aggregate.evaluateModel(classifier, test);
                    return null;
                });
                completedFolds++;
            } catch (IllegalStateException error) {
                if (remainingSeconds(deadline) > 0) {
                    throw error;
                }
                partial = true;
                lastError = errorMessage(error);
                break;
            }
        }
        SearchOutcome finalOutcome = evaluateSuccessiveHalving(data, candidates, arguments, "nested_cv",
                autoPreprocess, optimize, seed, true, deadline);
        String selectionSource = "full_data";
        if (finalOutcome.selected == null && fallback != null) {
            finalOutcome = new SearchOutcome(fallback, finalOutcome.candidates, finalOutcome.lastError,
                    true, finalOutcome.completed, finalOutcome.stages);
            selectionSource = "outer_training_fold";
        }
        completedEvaluations += finalOutcome.completed;
        partial |= finalOutcome.partial;
        if (finalOutcome.lastError != null) {
            lastError = finalOutcome.lastError;
        }
        for (JsonNode candidate : outerCandidates) {
            finalOutcome.candidates.add(candidate);
        }
        ObjectNode metrics = completedFolds == 0 ? null : metrics(aggregate, data);
        if (metrics != null) {
            metrics.put("outer_folds", outerFolds);
            metrics.put("completed_outer_folds", completedFolds);
            metrics.put("inner_folds", innerFolds);
            metrics.put("fold_safe_preprocessing", true);
            metrics.put("selection_scope", "outer_training_only");
            metrics.put("nested_cv_complete", !partial && completedFolds == outerFolds);
            metrics.put("partial", partial);
        }
        return new NestedHalvingResult(finalOutcome, metrics, outerResults, completedEvaluations, partial, lastError,
                selectionSource);
    }

    private record NestedHalvingResult(SearchOutcome outcome, ObjectNode testMetrics, ArrayNode outerFolds,
            int completedEvaluations, boolean partial, String lastError, String selectionSource) {
    }

    private record SearchOutcome(CandidateEvaluation selected, ArrayNode candidates, String lastError,
            boolean partial, int completed, ArrayNode stages) {
    }

    private CandidateEvaluation evaluateCandidate(Instances data, SearchCandidate candidate, Map<String, Object> arguments,
            String protocol, boolean autoPreprocess, String optimize, long seed, long deadline) throws Exception {
        return switch (protocol) {
            case "holdout" -> evaluateHoldoutValidation(data, candidate, arguments, autoPreprocess, optimize, seed, deadline);
            case "nested_cv" -> evaluateNestedValidation(data, candidate, arguments, autoPreprocess, optimize, seed, deadline);
            case "legacy_cv" -> evaluateLegacyValidation(data, candidate, arguments, autoPreprocess, optimize, seed, deadline);
            default -> throw new IllegalArgumentException("Unknown protocol: " + protocol);
        };
    }

    private CandidateEvaluation evaluateHoldoutValidation(Instances data, SearchCandidate candidate,
            Map<String, Object> arguments, boolean autoPreprocess, String optimize, long seed, long deadline)
            throws Exception {
        WekaTools.DataSplit split = split(arguments, data, seed);
        int folds = intValue(arguments, "folds", WekaTools.automaticFoldCount(split.training()));
        if (folds < 2 || folds > split.training().numInstances()) {
            throw new IllegalArgumentException("folds must be between 2 and the outer training instance count");
        }
        Classifier classifier = newClassifier(candidate, autoPreprocess, split.training());
        Evaluation evaluation = new Evaluation(split.training());
        call(deadline, () -> {
            evaluation.crossValidateModel(classifier, split.training(), folds, new Random(seed));
            return null;
        });
        ObjectNode metrics = metrics(evaluation, split.training());
        metrics.put("folds", folds);
        return new CandidateEvaluation(candidate, score(evaluation, split.training(), optimize), metrics);
    }

    private CandidateEvaluation evaluateNestedValidation(Instances data, SearchCandidate candidate,
            Map<String, Object> arguments, boolean autoPreprocess, String optimize, long seed, long deadline)
            throws Exception {
        int outerFolds = intValue(arguments, "outer_folds", Math.min(5, WekaTools.automaticFoldCount(data)));
        int innerFolds = intValue(arguments, "inner_folds", WekaTools.automaticFoldCount(data));
        if (outerFolds < 2 || outerFolds > data.numInstances() || innerFolds < 2) {
            throw new IllegalArgumentException("Invalid outer_folds or inner_folds");
        }
        Instances randomized = new Instances(data);
        randomized.randomize(new Random(seed));
        if (data.classAttribute().isNominal()) {
            randomized.stratify(outerFolds);
        }
        double weightedScore = 0;
        int total = 0;
        ArrayNode foldScores = ToolSupport.array();
        for (int fold = 0; fold < outerFolds; fold++) {
            Instances training = randomized.trainCV(outerFolds, fold);
            if (innerFolds > training.numInstances()) {
                throw new IllegalArgumentException("inner_folds exceeds an outer training fold size");
            }
            Classifier classifier = newClassifier(candidate, autoPreprocess, training);
            Evaluation evaluation = new Evaluation(training);
            long foldSeed = seed + fold;
            call(deadline, () -> {
                evaluation.crossValidateModel(classifier, training, innerFolds, new Random(foldSeed));
                return null;
            });
            double score = score(evaluation, training, optimize);
            foldScores.add(score);
            weightedScore += score * training.numInstances();
            total += training.numInstances();
        }
        ObjectNode metrics = ToolSupport.object();
        ToolSupport.putNumber(metrics, "mean_validation_score", weightedScore / total);
        metrics.set("outer_fold_scores", foldScores);
        return new CandidateEvaluation(candidate, weightedScore / total, metrics);
    }

    private CandidateEvaluation evaluateLegacyValidation(Instances data, SearchCandidate candidate,
            Map<String, Object> arguments, boolean autoPreprocess, String optimize, long seed, long deadline)
            throws Exception {
        int folds = intValue(arguments, "folds", WekaTools.automaticFoldCount(data));
        if (folds < 2 || folds > data.numInstances()) {
            throw new IllegalArgumentException("folds must be between 2 and the number of instances");
        }
        Classifier classifier = newClassifier(candidate, autoPreprocess, data);
        Evaluation evaluation = new Evaluation(data);
        call(deadline, () -> {
            evaluation.crossValidateModel(classifier, data, folds, new Random(seed));
            return null;
        });
        ObjectNode metrics = metrics(evaluation, data);
        return new CandidateEvaluation(candidate, score(evaluation, data, optimize), metrics);
    }

    private ObjectNode evaluateTest(Instances data, SearchCandidate candidate, Map<String, Object> arguments,
            String protocol, boolean autoPreprocess, long seed, long deadline) throws Exception {
        if ("legacy_cv".equals(protocol)) {
            return null;
        }
        if ("holdout".equals(protocol)) {
            WekaTools.DataSplit split = split(arguments, data, seed);
            Classifier classifier = newClassifier(candidate, autoPreprocess, split.training());
            call(deadline, () -> {
                classifier.buildClassifier(split.training());
                return null;
            });
            Evaluation evaluation = new Evaluation(split.training());
            call(deadline, () -> {
                evaluation.evaluateModel(classifier, split.test());
                return null;
            });
            return metrics(evaluation, split.test());
        }
        int outerFolds = intValue(arguments, "outer_folds", Math.min(5, WekaTools.automaticFoldCount(data)));
        Instances randomized = new Instances(data);
        randomized.randomize(new Random(seed));
        if (data.classAttribute().isNominal()) {
            randomized.stratify(outerFolds);
        }
        Evaluation aggregate = new Evaluation(data);
        for (int fold = 0; fold < outerFolds; fold++) {
            Instances training = randomized.trainCV(outerFolds, fold);
            Instances test = randomized.testCV(outerFolds, fold);
            Classifier classifier = newClassifier(candidate, autoPreprocess, training);
            call(deadline, () -> {
                classifier.buildClassifier(training);
                return null;
            });
            call(deadline, () -> {
                aggregate.evaluateModel(classifier, test);
                return null;
            });
        }
        return metrics(aggregate, data);
    }

    private WekaTools.DataSplit split(Map<String, Object> arguments, Instances data, long seed) {
        double trainPercent = doubleValue(arguments, "train_percent", 80.0);
        if (trainPercent <= 0 || trainPercent >= 100) {
            throw new IllegalArgumentException("train_percent must be between 0 and 100");
        }
        return WekaTools.splitData(data, trainPercent, seed);
    }

    private Classifier newClassifier(SearchCandidate candidate, boolean autoPreprocess, Instances data) {
        List<FilterSpec> preprocessing = autoPreprocess
                ? Preprocessing.automatic(data, candidate.classifierClass())
                : List.of();
        return newClassifier(candidate, preprocessing);
    }

    private Classifier newClassifier(SearchCandidate candidate, List<FilterSpec> preprocessing) {
        Classifier base = (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER,
                candidate.classifierClass(), candidate.options());
        return Preprocessing.wrap(base, preprocessing, whitelist);
    }

    private <T> T call(long deadline, Callable<T> task) throws Exception {
        long remaining = remainingSeconds(deadline);
        if (remaining <= 0) {
            throw new IllegalStateException("AutoML budget exhausted");
        }
        return tasks.call(Math.min(config.trainTimeoutSeconds(), remaining), task);
    }

    private static long remainingSeconds(long deadline) {
        return Math.max(0, (deadline - System.nanoTime() + 999_999_999L) / 1_000_000_000L);
    }

    private static ObjectNode metrics(Evaluation evaluation, Instances data) throws Exception {
        ObjectNode result = ToolSupport.object();
        WekaTools.addEvaluationMetrics(result, evaluation, data);
        return result;
    }

    private static double score(ObjectNode metrics, Instances data, String optimize) {
        if (metrics == null) {
            return Double.NaN;
        }
        return switch (optimize) {
            case "rmse" -> metrics.get("rmse").doubleValue();
            case "weighted_f1" -> weightedF1(metrics, data);
            default -> metrics.get("accuracy_percent").doubleValue();
        };
    }

    private static double score(Evaluation evaluation, Instances data, String optimize) {
        return switch (optimize) {
            case "rmse" -> evaluation.rootMeanSquaredError();
            case "weighted_f1" -> weightedF1(evaluation, data);
            default -> evaluation.pctCorrect();
        };
    }

    private static double weightedF1(Evaluation evaluation, Instances data) {
        double total = 0;
        double weighted = 0;
        for (int index = 0; index < data.classAttribute().numValues(); index++) {
            double support = evaluation.numTruePositives(index) + evaluation.numFalseNegatives(index);
            double f1 = evaluation.fMeasure(index);
            if (Double.isFinite(f1)) {
                weighted += support * f1;
                total += support;
            }
        }
        return total == 0 ? Double.NaN : weighted / total;
    }

    private static double weightedF1(ObjectNode metrics, Instances data) {
        if (!metrics.has("per_class")) {
            return Double.NaN;
        }
        double total = 0;
        double weighted = 0;
        for (JsonNode item : metrics.get("per_class")) {
            if (item.has("f1") && item.has("support")) {
                weighted += item.get("support").doubleValue() * item.get("f1").doubleValue();
                total += item.get("support").doubleValue();
            }
        }
        return total == 0 ? Double.NaN : weighted / total;
    }

    private record CandidateEvaluation(SearchCandidate candidate, double validationScore,
            ObjectNode validationMetrics) {
    }

    private static String required(Map<String, Object> arguments, String name) {
        return ToolSupport.requiredString(arguments, name);
    }

    private static String stringValue(Map<String, Object> arguments, String name, String defaultValue) {
        return ToolSupport.optionalString(arguments, name, defaultValue);
    }

    private static int intValue(Map<String, Object> arguments, String name, int defaultValue) {
        return ToolSupport.optionalInt(arguments, name, defaultValue);
    }

    private static long longValue(Map<String, Object> arguments, String name, long defaultValue) {
        return ToolSupport.optionalLong(arguments, name, defaultValue);
    }

    private static double doubleValue(Map<String, Object> arguments, String name, double defaultValue) {
        return ToolSupport.optionalDouble(arguments, name, defaultValue);
    }

    private static boolean booleanValue(Map<String, Object> arguments, String name, boolean defaultValue) {
        return ToolSupport.optionalBoolean(arguments, name, defaultValue);
    }
}
