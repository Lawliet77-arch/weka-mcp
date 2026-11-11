package dev.wekamcp.tools;

import dev.wekamcp.analysis.AnalysisService;
import dev.wekamcp.automl.AutoMlService;
import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.experiment.ExperimentRecord;
import dev.wekamcp.experiment.ExperimentStore;
import dev.wekamcp.experiment.ModelCodec;
import dev.wekamcp.engine.Preprocessing;
import dev.wekamcp.engine.Preprocessing.FilterSpec;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.ClustererEntry;
import dev.wekamcp.registry.SessionRegistry.DatasetEntry;
import dev.wekamcp.registry.SessionRegistry.ModelEntry;
import io.modelcontextprotocol.server.McpServerFeatures.SyncToolSpecification;
import io.modelcontextprotocol.spec.McpSchema;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

import static dev.wekamcp.tools.ToolSupport.array;
import static dev.wekamcp.tools.ToolSupport.object;

public final class WekaTools {
    private static final Logger LOGGER = LoggerFactory.getLogger(WekaTools.class);
    private static final String OUTPUT_PLACEHOLDER_ID = "00000000-0000-0000-0000-000000000000";
    private static final Set<String> AUDIT_KEYS = Set.of(
            "dataset_id", "model_id", "clusterer_id", "resource_id", "filter_class", "classifier_class",
            "clusterer_class", "options", "class_index", "mode", "folds", "seed", "num_clusters",
            "auto_preprocess", "test_dataset_id", "remove_class", "evaluate");

    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;
    private final ExperimentStore experiments;
    private final AutoMlService autoMl;
    private final AnalysisService analysis;
    private final dev.wekamcp.engine.StreamingService streaming;
    private final dev.wekamcp.interop.ModelInteropService interop;
    private final dev.wekamcp.workflow.WorkflowService workflow;
    private final dev.wekamcp.evaluation.ModelComparisonService comparison;

    public WekaTools(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks)
            throws java.io.IOException {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
        this.experiments = new ExperimentStore(config);
        this.autoMl = new AutoMlService(config, whitelist, registry, tasks, experiments);
        this.analysis = new AnalysisService(config, whitelist, registry, tasks);
        this.streaming = new dev.wekamcp.engine.StreamingService(config, whitelist, registry, tasks, experiments);
        this.interop = new dev.wekamcp.interop.ModelInteropService(config, whitelist, registry, tasks, experiments);
        this.workflow = new dev.wekamcp.workflow.WorkflowService(config, whitelist, registry, tasks);
        this.comparison = new dev.wekamcp.evaluation.ModelComparisonService(config, whitelist, registry, tasks);
    }

    public List<SyncToolSpecification> specifications() {
        return List.of(
                specification("weka_load_dataset", "Load an ARFF or CSV file from the configured data root",
                        loadSchema(), this::loadDataset),
                specification("weka_describe_dataset", "Describe dataset attributes, types, nominal values and class index",
                        idSchema("dataset_id"), this::describeDataset),
                specification("weka_list_resources", "List datasets, trained models and clusterers in this session",
                        ToolSupport.schema(Map.of()), this::listResources),
                specification("weka_list_algorithms", "Discover whitelisted algorithms, options, capabilities and update support",
                        listAlgorithmsSchema(), this::listAlgorithms),
                specification("weka_recommend", "Recommend task-compatible classifiers without claiming unvalidated performance",
                        recommendSchema(), this::recommend),
                specification("weka_list_filters", "List whitelisted Weka filters and their allowed options",
                        ToolSupport.schema(Map.of()), this::listFilters),
                specification("weka_apply_filter", "Apply a whitelisted filter and register a new dataset",
                        filterSchema(), this::applyFilter),
                specification("weka_split_dataset", "Create a reproducible stratified training and test split",
                        splitDatasetSchema(), this::splitDataset),
                specification("weka_transform_preview", "Preview a full-dataset filter transformation without registering it",
                        previewSchema(), this::transformPreview),
                specification("weka_histogram", "Return bounded numeric or nominal histogram bins",
                        histogramSchema(), this::histogram),
                specification("weka_scatter", "Return a bounded numeric scatter plot with optional class labels",
                        scatterSchema(), this::scatter),
                specification("weka_scatter_matrix", "Return bounded pairwise scatter plots for up to five numeric attributes",
                        scatterMatrixSchema(), this::scatterMatrix),
                specification("weka_compare_datasets", "Compare dataset schemas and descriptive distribution drift",
                        compareDatasetsSchema(), this::compareDatasets),
                specification("weka_export_dataset", "Export a session dataset as ARFF or CSV inside the data root",
                        exportDatasetSchema(), this::exportDataset),
                specification("weka_profile_dataset", "Profile missing values, distributions, duplicates, imbalance and correlations",
                        idSchema("dataset_id"), this::profileDataset),
                specification("weka_learning_curve", "Evaluate a classifier across increasing training fractions",
                        learningCurveSchema(), this::learningCurve),
                specification("weka_model_diagnostics", "Inspect classification curves, calibration, residuals and errors",
                        modelDiagnosticsSchema(), this::modelDiagnostics),
                specification("weka_threshold_analysis", "Sweep binary classification thresholds, costs and margin distributions",
                        thresholdAnalysisSchema(), this::thresholdAnalysis),
                specification("weka_select_attributes", "Rank or select attributes with whitelisted evaluators",
                        selectAttributesSchema(), this::selectAttributes),
                specification("weka_associate", "Mine bounded Apriori or FPGrowth association rules",
                        associateSchema(), this::associate),
                specification("weka_explain_model", "Return J48 text tree and DOT when supported",
                        idSchema("model_id"), this::explainModel),
                specification("weka_train_classifier", "Train a whitelisted classifier and register a reusable model",
                        trainSchema(), this::trainClassifier),
                specification("weka_train_stream", "Incrementally train an updateable classifier from an ARFF or CSV stream",
                        trainStreamSchema(), this::trainStream),
                specification("weka_auto_classify", "Automatically preprocess, evaluate and train a final classifier",
                        autoClassifySchema(), this::autoClassify),
                specification("weka_evaluate", "Evaluate a classifier using cross-validation or a stratified holdout split",
                        evaluateSchema(), this::evaluate),
                specification("weka_compare_models", "Compare models on shared repeated folds with corrected paired significance tests",
                        compareModelsSchema(), this::compareModels),
                specification("weka_evaluate_model", "Evaluate an existing trained model on an independent labeled test dataset",
                        evaluateModelSchema(), this::evaluateModel),
                specification("weka_list_experiments", "List persistent experiments",
                        listExperimentsSchema(), this::listExperiments),
                specification("weka_get_experiment", "Get a persistent experiment manifest",
                        experimentIdSchema(), this::getExperiment),
                specification("weka_load_experiment_model", "Restore a model from a persistent experiment",
                        experimentIdSchema(), this::loadExperimentModel),
                specification("weka_save_model", "Persist a trained classifier with its training data",
                        idSchema("model_id"), this::saveModel),
                specification("weka_save_clusterer", "Persist a trained clusterer with its training data",
                        idSchema("clusterer_id"), this::saveClusterer),
                specification("weka_load_clusterer", "Restore a clusterer from a persistent experiment",
                        experimentIdSchema(), this::loadClusterer),
                specification("weka_export_model", "Export a raw Weka model or GUI-compatible model and training header",
                        exportModelSchema(), this::exportModel),
                specification("weka_import_model", "Import a trusted Weka model with type filtering and header validation",
                        importModelSchema(), this::importModel),
                specification("weka_update_model", "Incrementally update an updateable classifier from a dataset or file",
                        updateModelSchema(), this::updateModel),
                specification("weka_compare_experiments", "Compare persistent experiments by validation or test score",
                        compareExperimentsSchema(), this::compareExperiments),
                specification("weka_delete_experiment", "Delete a persistent experiment",
                        experimentIdSchema(), this::deleteExperiment),
                specification("weka_predict", "Predict a registered dataset or one inline JSON instance with a trained model",
                        predictSchema(), this::predict),
                specification("weka_cluster", "Train a whitelisted clusterer and return cluster assignments",
                        clusterSchema(), this::cluster),
                specification("weka_cluster_predict", "Assign new instances to an existing clusterer",
                        clusterPredictSchema(), this::clusterPredict),
                specification("weka_cluster_evaluate", "Evaluate cluster sizes, class mappings, likelihood and sampled silhouette",
                        clusterEvaluateSchema(), this::clusterEvaluate),
                specification("weka_release", "Release a dataset, model or clusterer by its UUID",
                        idSchema("resource_id"), this::release));
    }

    public CallToolResult loadDataset(Map<String, Object> arguments) {
        return handle("weka_load_dataset", arguments, () -> {
            Path path = config.resolveDatasetPath(ToolSupport.requiredString(arguments, "path"));
            String format = ToolSupport.optionalString(arguments, "format", "auto");
            if (!Set.of("auto", "arff", "csv").contains(format)) {
                throw new IllegalArgumentException("format must be auto, arff or csv");
            }
            Instances data = tasks.call(config.loadTimeoutSeconds(), () -> {
                if ("csv".equals(format)) {
                    CSVLoader loader = new CSVLoader();
                    loader.setFile(path.toFile());
                    return loader.getDataSet();
                }
                if ("arff".equals(format)) {
                    weka.core.converters.ArffLoader loader = new weka.core.converters.ArffLoader();
                    loader.setFile(path.toFile());
                    return loader.getDataSet();
                }
                return new DataSource(path.toString()).getDataSet();
            });
            validateDataset(data);
            int classIndex = ToolSupport.optionalInt(arguments, "class_index", 0);
            setRequestedClassIndex(data, classIndex);
            String defaultName = ToolSupport.optionalString(arguments, "name", path.getFileName().toString());
            ensureOutputWithinLimit(datasetSummary(new DatasetEntry(OUTPUT_PLACEHOLDER_ID, defaultName, data)));
            DatasetEntry entry = registry.registerDataset(defaultName, data);
            return datasetSummary(entry);
        });
    }

    public CallToolResult describeDataset(Map<String, Object> arguments) {
        return handle("weka_describe_dataset", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            ObjectNode result = datasetSummary(entry);
            ArrayNode attributes = result.putArray("attribute_details");
            for (int index = 0; index < entry.data().numAttributes(); index++) {
                Attribute attribute = entry.data().attribute(index);
                ObjectNode item = attributes.addObject();
                item.put("index", index + 1);
                item.put("name", attribute.name());
                item.put("type", attributeType(attribute));
                item.put("is_class", index == entry.data().classIndex());
                if (attribute.isNominal()) {
                    ArrayNode values = item.putArray("nominal_values");
                    for (int value = 0; value < attribute.numValues(); value++) {
                        values.add(attribute.value(value));
                    }
                }
            }
            return result;
        });
    }

    public CallToolResult listAlgorithms(Map<String, Object> arguments) {
        return handle("weka_list_algorithms", arguments, () -> {
            String requested = ToolSupport.optionalString(arguments, "kind", "all");
            if (!Set.of("all", "classifier", "filter", "clusterer", "associator").contains(requested)) {
                throw new IllegalArgumentException("kind must be all, classifier, filter, clusterer or associator");
            }
            ObjectNode result = object();
            result.put("kind", requested);
            int total = 0;
            for (Whitelist.Kind kind : Whitelist.Kind.values()) {
                if (!"all".equals(requested) && !kind.name().toLowerCase(java.util.Locale.ROOT).equals(requested)) {
                    continue;
                }
                ArrayNode algorithms = result.putArray(kind.name().toLowerCase(java.util.Locale.ROOT) + "s");
                for (Whitelist.AlgorithmSpec specification : whitelist.list(kind)) {
                    Object instance = whitelist.create(kind, specification.className(), new String[0]);
                    ObjectNode item = algorithms.addObject();
                    item.put("class", specification.className());
                    item.set("allowed_options", ToolSupport.JSON.valueToTree(specification.allowedOptions()));
                    if (instance instanceof weka.classifiers.UpdateableClassifier
                            || instance instanceof weka.clusterers.UpdateableClusterer) {
                        item.put("incremental", true);
                    } else {
                        item.put("incremental", false);
                    }
                    if (instance instanceof weka.core.CapabilitiesHandler handler) {
                        weka.core.Capabilities capabilities = handler.getCapabilities();
                        ArrayNode supported = item.putArray("capabilities");
                        capabilities.capabilities().forEachRemaining(capability -> supported.add(capability.name()));
                        item.put("minimum_instances", capabilities.getMinimumNumberInstances());
                    }
                }
                total += algorithms.size();
            }
            result.put("count", total);
            return result;
        });
    }

    public CallToolResult recommend(Map<String, Object> arguments) {
        return handle("weka_recommend", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            Instances data = classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
            boolean streaming = ToolSupport.optionalBoolean(arguments, "streaming", false);
            int limit = ToolSupport.optionalInt(arguments, "limit", 5);
            if (limit < 1 || limit > 20) {
                throw new IllegalArgumentException("limit must be between 1 and 20");
            }
            weka.core.Capabilities required = weka.core.Capabilities.forInstances(data);
            List<ObjectNode> candidates = new ArrayList<>();
            List<String> defaults = defaultCandidates(data);
            for (Whitelist.AlgorithmSpec specification : whitelist.list(Whitelist.Kind.CLASSIFIER)) {
                Classifier classifier = (Classifier) whitelist.create(
                        Whitelist.Kind.CLASSIFIER, specification.className(), new String[0]);
                boolean incremental = classifier instanceof weka.classifiers.UpdateableClassifier;
                if (streaming && !incremental || !classifier.getCapabilities().supports(required)) {
                    continue;
                }
                int priority = defaults.indexOf(specification.className());
                int score = priority < 0 ? 0 : defaults.size() - priority;
                if (incremental) {
                    score++;
                }
                ObjectNode item = object();
                item.put("classifier_class", specification.className());
                item.put("incremental", incremental);
                item.put("score", score);
                item.put("reason", "Compatible with the requested class and attribute types"
                        + (incremental ? "; supports incremental updates" : ""));
                item.set("options", ToolSupport.JSON.valueToTree(specification.allowedOptions()));
                candidates.add(item);
            }
            candidates.sort((left, right) -> {
                int score = Integer.compare(right.get("score").intValue(), left.get("score").intValue());
                return score != 0 ? score : left.get("classifier_class").asText()
                        .compareTo(right.get("classifier_class").asText());
            });
            ObjectNode result = object();
            result.put("dataset_id", entry.id());
            result.put("class_index", data.classIndex() + 1);
            result.put("streaming", streaming);
            result.put("ranking", "capability_and_complexity_heuristic");
            result.put("requires_validation", true);
            result.put("recommended_next_tool", streaming ? "weka_train_stream" : "weka_auto_classify");
            result.put("candidate_count", candidates.size());
            ArrayNode recommendations = result.putArray("recommendations");
            candidates.stream().limit(limit).forEach(recommendations::add);
            return result;
        });
    }

    public CallToolResult listFilters(Map<String, Object> arguments) {
        return handle("weka_list_filters", arguments, () -> {
            ObjectNode result = object();
            result.set("filters", ToolSupport.JSON.valueToTree(whitelist.list(Whitelist.Kind.FILTER)));
            return result;
        });
    }

    public CallToolResult splitDataset(Map<String, Object> arguments) {
        return handle("weka_split_dataset", arguments, () -> workflow.split(arguments));
    }

    public CallToolResult transformPreview(Map<String, Object> arguments) {
        return handle("weka_transform_preview", arguments, () -> workflow.preview(arguments));
    }

    public CallToolResult histogram(Map<String, Object> arguments) {
        return handle("weka_histogram", arguments, () -> workflow.histogram(arguments));
    }

    public CallToolResult scatter(Map<String, Object> arguments) {
        return handle("weka_scatter", arguments, () -> workflow.scatter(arguments));
    }

    public CallToolResult scatterMatrix(Map<String, Object> arguments) {
        return handle("weka_scatter_matrix", arguments, () -> workflow.scatterMatrix(arguments));
    }

    public CallToolResult compareDatasets(Map<String, Object> arguments) {
        return handle("weka_compare_datasets", arguments, () -> workflow.compare(arguments));
    }

    public CallToolResult exportDataset(Map<String, Object> arguments) {
        return handle("weka_export_dataset", arguments, () -> workflow.export(arguments));
    }

    public CallToolResult applyFilter(Map<String, Object> arguments) {
        return handle("weka_apply_filter", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            String className = ToolSupport.requiredString(arguments, "filter_class");
            String[] options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
            Filter filter = (Filter) whitelist.create(Whitelist.Kind.FILTER, className, options);
            Instances input = new Instances(entry.data());
            setRequestedClassIndex(input, ToolSupport.optionalInt(arguments, "class_index", 0));
            if (filter instanceof SupervisedFilter && input.classIndex() < 0) {
                throw new IllegalArgumentException("A supervised filter requires class_index");
            }
            Instances output = tasks.call(config.loadTimeoutSeconds(), () -> {
                filter.setInputFormat(input);
                return Filter.useFilter(input, filter);
            });
            validateDataset(output);
            String simpleName = className.substring(className.lastIndexOf('.') + 1);
            String resultName = entry.name() + "." + simpleName;
            ObjectNode result = datasetSummary(new DatasetEntry(OUTPUT_PLACEHOLDER_ID, resultName, output));
            result.put("source_dataset_id", entry.id());
            result.put("filter_class", className);
            ensureOutputWithinLimit(result);
            DatasetEntry filtered = registry.registerDataset(resultName, output);
            result.put("dataset_id", filtered.id());
            return result;
        });
    }

    public CallToolResult trainStream(Map<String, Object> arguments) {
        return handle("weka_train_stream", arguments, () -> streaming.train(arguments));
    }

    public CallToolResult trainClassifier(Map<String, Object> arguments) {
        return handle("weka_train_classifier", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            String className = ToolSupport.requiredString(arguments, "classifier_class");
            String[] options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
            Instances training = classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
            List<FilterSpec> preprocessing = ToolSupport.optionalBoolean(arguments, "auto_preprocess", false)
                    ? Preprocessing.automatic(training, className)
                    : List.of();
            Classifier base = (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER, className, options);
            Classifier classifier = Preprocessing.wrap(base, preprocessing, whitelist);
            long started = System.nanoTime();
            tasks.call(config.trainTimeoutSeconds(), () -> {
                classifier.buildClassifier(training);
                return null;
            });
            ModelEntry trained = new ModelEntry(
                    OUTPUT_PLACEHOLDER_ID, className, options, classifier, training, preprocessing);
            ModelEntry model;
            if (ToolSupport.optionalBoolean(arguments, "persist", false)) {
                ExperimentRecord saved = persistModel(trained);
                model = registry.registerModel(className, options, classifier, training, preprocessing,
                        training, saved.id);
            } else {
                model = registry.registerModel(className, options, classifier, training, preprocessing);
            }
            ObjectNode result = modelSummary(model);
            result.put("training_ms", (System.nanoTime() - started) / 1_000_000);
            return result;
        });
    }

    public CallToolResult evaluate(Map<String, Object> arguments) {
        return handle("weka_evaluate", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            String modelId = ToolSupport.optionalString(arguments, "model_id", null);
            String requestedClass = ToolSupport.optionalString(arguments, "classifier_class", null);
            if ((modelId == null) == (requestedClass == null)) {
                throw new IllegalArgumentException("Specify exactly one of model_id or classifier_class");
            }
            if (modelId != null && arguments.containsKey("options")) {
                throw new IllegalArgumentException("options must not be provided with model_id; saved model options are reused");
            }
            Instances data = classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
            String className;
            String[] options;
            List<FilterSpec> preprocessing;
            if (modelId != null) {
                ModelEntry model = registry.model(modelId);
                if (!model.retrainable()) {
                    throw new IllegalArgumentException(
                            "Imported models preserve their original options; use weka_evaluate_model for direct evaluation");
                }
                className = model.classifierClass();
                options = model.options();
                preprocessing = model.preprocessing();
            } else {
                className = requestedClass;
                options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
                preprocessing = ToolSupport.optionalBoolean(arguments, "auto_preprocess", false)
                        ? Preprocessing.automatic(data, className)
                        : List.of();
            }
            Classifier base = (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER, className, options);
            Classifier classifier = Preprocessing.wrap(base, preprocessing, whitelist);
            String mode = ToolSupport.optionalString(arguments, "mode", "cross_validation");
            long seed = ToolSupport.optionalLong(arguments, "seed", 1L);
            int folds = ToolSupport.optionalInt(arguments, "folds", 10);
            double trainPercent = ToolSupport.optionalDouble(arguments, "train_percent", 66.7);
            int repeats = ToolSupport.optionalInt(arguments, "repeats", 1);
            if (!Set.of("cross_validation", "holdout").contains(mode)) {
                throw new IllegalArgumentException("mode must be cross_validation or holdout");
            }
            if ("cross_validation".equals(mode) && (folds < 2 || folds > data.numInstances())) {
                throw new IllegalArgumentException("folds must be between 2 and the number of instances");
            }
            if ("holdout".equals(mode) && (trainPercent <= 0 || trainPercent >= 100)) {
                throw new IllegalArgumentException("train_percent must be between 0 and 100");
            }
            if (repeats < 1 || repeats > 20) {
                throw new IllegalArgumentException("repeats must be between 1 and 20");
            }
            ObjectNode result = object();
            result.put("dataset_id", entry.id());
            result.put("classifier_class", className);
            result.put("mode", mode);
            result.put("seed", seed);
            result.put("instances", data.numInstances());
            result.put("class_type", data.classAttribute().isNominal() ? "nominal" : "numeric");
            result.put("repeats", repeats);
            result.set("preprocessing", ToolSupport.JSON.valueToTree(preprocessing));
            tasks.call(config.trainTimeoutSeconds(), () -> {
                if (repeats > 1) {
                    if ("cross_validation".equals(mode)) {
                        result.put("folds", folds);
                    } else {
                        DataSplit split = splitData(data, trainPercent, seed);
                        result.put("train_percent", trainPercent);
                        result.put("training_instances", split.training().numInstances());
                        result.put("test_instances", split.test().numInstances());
                        result.put("stratified", data.classAttribute().isNominal());
                    }
                    dev.wekamcp.evaluation.ModelComparisonService.addRepeatedMetrics(result,
                            comparison.repeat(data, classifier, mode, folds, trainPercent, seed, repeats), data);
                    return null;
                }
                Evaluation evaluation;
                if ("cross_validation".equals(mode)) {
                    evaluation = new Evaluation(data);
                    evaluation.crossValidateModel(classifier, data, folds, new Random(seed));
                    result.put("folds", folds);
                } else {
                    DataSplit split = splitData(data, trainPercent, seed);
                    Instances training = split.training();
                    Instances test = split.test();
                    classifier.buildClassifier(training);
                    evaluation = new Evaluation(training);
                    evaluation.evaluateModel(classifier, test);
                    result.put("train_percent", trainPercent);
                    result.put("training_instances", training.numInstances());
                    result.put("test_instances", test.numInstances());
                    result.put("stratified", data.classAttribute().isNominal());
                }
                addEvaluationMetrics(result, evaluation, data);
                return null;
            });
            return result;
        });
    }

    public CallToolResult autoClassify(Map<String, Object> arguments) {
        return handle("weka_auto_classify", arguments, () -> autoMl.autoClassify(arguments));
    }

    private CallToolResult legacyAutoClassify(Map<String, Object> arguments) {
        return handle("weka_auto_classify", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            String requestedClass = ToolSupport.optionalString(arguments, "classifier_class", "auto");
            boolean automatic = "auto".equals(requestedClass);
            if (automatic && arguments.containsKey("options")) {
                throw new IllegalArgumentException("options requires an explicit classifier_class because candidate options differ");
            }
            boolean autoPreprocess = ToolSupport.optionalBoolean(arguments, "auto_preprocess", true);
            boolean runEvaluation = ToolSupport.optionalBoolean(arguments, "evaluate", true);
            Instances classified = classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
            if (runEvaluation && classified.numInstances() < 2) {
                throw new IllegalArgumentException("Automatic evaluation requires at least two instances");
            }

            Map<String, Object> baseArguments = new LinkedHashMap<>();
            for (String key : List.of("dataset_id", "options", "class_index")) {
                if (arguments.containsKey(key)) {
                    baseArguments.put(key, arguments.get(key));
                }
            }
            baseArguments.put("auto_preprocess", autoPreprocess);
            List<String> candidates = automatic && runEvaluation
                    ? defaultCandidates(classified)
                    : List.of(automatic ? "weka.classifiers.trees.RandomForest" : requestedClass);
            boolean nominal = classified.classAttribute().isNominal();
            String metric = nominal ? "accuracy_percent" : "rmse";
            String selectedClass = null;
            JsonNode selectedEvaluation = null;
            double selectedScore = Double.NaN;
            String lastError = null;
            ArrayNode candidateResults = array();
            for (String candidate : candidates) {
                JsonNode evaluation = null;
                if (runEvaluation) {
                    Map<String, Object> evaluationArguments = new LinkedHashMap<>(baseArguments);
                    evaluationArguments.put("classifier_class", candidate);
                    for (String key : List.of("mode", "folds", "seed", "train_percent")) {
                        if (arguments.containsKey(key)) {
                            evaluationArguments.put(key, arguments.get(key));
                        }
                    }
                    String mode = ToolSupport.optionalString(evaluationArguments, "mode", "cross_validation");
                    if ("cross_validation".equals(mode) && !evaluationArguments.containsKey("folds")) {
                        evaluationArguments.put("folds", automaticFoldCount(classified));
                    }
                    ObjectNode candidateResult = candidateResults.addObject();
                    candidateResult.put("classifier_class", candidate);
                    try {
                        evaluation = payloadOf(evaluate(evaluationArguments));
                        double score = evaluation.get(metric).doubleValue();
                        candidateResult.put("status", "OK");
                        ToolSupport.putNumber(candidateResult, "score", score);
                        candidateResult.set("evaluation", evaluation);
                        boolean better = selectedEvaluation == null
                                || (nominal ? score > selectedScore : score < selectedScore);
                        if (better) {
                            selectedClass = candidate;
                            selectedEvaluation = evaluation;
                            selectedScore = score;
                        }
                    } catch (IllegalArgumentException e) {
                        if (!automatic) {
                            throw e;
                        }
                        lastError = e.getMessage();
                        candidateResult.put("status", "ERROR");
                        candidateResult.put("error", lastError);
                    }
                }
            }
            if (runEvaluation && selectedEvaluation == null) {
                throw new IllegalStateException("No candidate classifier could be evaluated" +
                        (lastError == null ? "" : ": " + lastError));
            }
            if (selectedClass == null) {
                selectedClass = candidates.get(0);
            }
            Map<String, Object> trainingArguments = new LinkedHashMap<>(baseArguments);
            trainingArguments.put("classifier_class", selectedClass);
            JsonNode model = payloadOf(trainClassifier(trainingArguments));
            ObjectNode result = object();
            result.put("model_id", model.get("model_id").asText());
            result.put("classifier_class", selectedClass);
            result.put("auto_selected", automatic && runEvaluation);
            result.put("selection_metric", runEvaluation ? metric : "default");
            ToolSupport.putNumber(result, "selection_score", selectedScore);
            result.put("auto_preprocess", autoPreprocess);
            result.set("preprocessing", model.get("preprocessing"));
            result.set("model", model);
            if (selectedEvaluation == null) {
                result.putNull("evaluation");
            } else {
                result.set("evaluation", selectedEvaluation);
            }
            result.set("candidates", candidateResults);
            return result;
        });
    }

    public CallToolResult profileDataset(Map<String, Object> arguments) {
        return handle("weka_profile_dataset", arguments, () -> analysis.profileDataset(
                ToolSupport.requiredString(arguments, "dataset_id")));
    }

    public CallToolResult learningCurve(Map<String, Object> arguments) {
        return handle("weka_learning_curve", arguments, () -> analysis.learningCurve(
                ToolSupport.requiredString(arguments, "dataset_id"),
                ToolSupport.optionalString(arguments, "classifier_class", "weka.classifiers.trees.RandomForest"),
                ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength()),
                ToolSupport.optionalInt(arguments, "class_index", 0),
                ToolSupport.optionalString(arguments, "fractions", "0.25,0.5,0.75,1.0"),
                ToolSupport.optionalLong(arguments, "seed", 1L),
                ToolSupport.optionalBoolean(arguments, "auto_preprocess", true)));
    }

    public CallToolResult modelDiagnostics(Map<String, Object> arguments) {
        return handle("weka_model_diagnostics", arguments, () -> analysis.modelDiagnostics(
                ToolSupport.requiredString(arguments, "model_id"),
                ToolSupport.requiredString(arguments, "dataset_id"),
                ToolSupport.optionalInt(arguments, "bins", 10),
                ToolSupport.optionalInt(arguments, "max_errors", 50)));
    }

    public CallToolResult thresholdAnalysis(Map<String, Object> arguments) {
        return handle("weka_threshold_analysis", arguments, () -> analysis.thresholdAnalysis(arguments));
    }

    public CallToolResult compareModels(Map<String, Object> arguments) {
        return handle("weka_compare_models", arguments, () -> comparison.compare(arguments));
    }

    public CallToolResult selectAttributes(Map<String, Object> arguments) {
        return handle("weka_select_attributes", arguments, () -> analysis.selectAttributes(
                ToolSupport.requiredString(arguments, "dataset_id"),
                ToolSupport.optionalInt(arguments, "class_index", 0),
                ToolSupport.optionalString(arguments, "method", "info_gain"),
                ToolSupport.optionalInt(arguments, "max_attributes", 10)));
    }

    public CallToolResult associate(Map<String, Object> arguments) {
        return handle("weka_associate", arguments, () -> analysis.associate(
                ToolSupport.requiredString(arguments, "dataset_id"),
                ToolSupport.optionalString(arguments, "algorithm", "apriori"),
                ToolSupport.optionalDouble(arguments, "min_support", 0.1),
                ToolSupport.optionalDouble(arguments, "min_confidence", 0.5),
                ToolSupport.optionalInt(arguments, "max_rules", 20),
                ToolSupport.optionalBoolean(arguments, "auto_prepare", true),
                ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength())));
    }

    public CallToolResult explainModel(Map<String, Object> arguments) {
        return handle("weka_explain_model", arguments, () -> analysis.explainModel(
                ToolSupport.requiredString(arguments, "model_id")));
    }

    public CallToolResult listExperiments(Map<String, Object> arguments) {
        return handle("weka_list_experiments", arguments, () -> {
            int offset = ToolSupport.optionalInt(arguments, "offset", 0);
            int limit = ToolSupport.optionalInt(arguments, "limit", 20);
            ObjectNode result = object();
            result.set("experiments", ToolSupport.JSON.valueToTree(experiments.list(offset, limit)));
            result.put("total", experiments.count());
            result.put("offset", offset);
            result.put("limit", limit);
            return result;
        });
    }

    public CallToolResult getExperiment(Map<String, Object> arguments) {
        return handle("weka_get_experiment", arguments, () -> {
            String id = ToolSupport.requiredString(arguments, "experiment_id");
            return ToolSupport.JSON.valueToTree(experiments.get(id));
        });
    }

    public CallToolResult exportModel(Map<String, Object> arguments) {
        return handle("weka_export_model", arguments, () -> interop.export(arguments));
    }

    public CallToolResult importModel(Map<String, Object> arguments) {
        return handle("weka_import_model", arguments, () -> interop.importModel(arguments));
    }

    public CallToolResult updateModel(Map<String, Object> arguments) {
        return handle("weka_update_model", arguments, () -> interop.update(arguments));
    }

    public CallToolResult loadExperimentModel(Map<String, Object> arguments) {
        return handle("weka_load_experiment_model", arguments, () -> {
            String id = ToolSupport.requiredString(arguments, "experiment_id");
            ExperimentRecord record = experiments.get(id);
            if (!"CLASSIFIER".equals(record.kind)) {
                throw new IllegalArgumentException("Experiment does not contain a classifier: " + id);
            }
            whitelist.get(Whitelist.Kind.CLASSIFIER, record.classifierClass);
            ModelCodec.ModelPayload payload = experiments.loadModel(id);
            Instances trainingData = record.sourcePath == null ? experiments.loadDataset(id) : null;
            ModelEntry model = record.retrainable
                    ? registry.registerModel(record.classifierClass, record.options, payload.classifier(),
                            payload.trainingHeader(), record.preprocessing, trainingData, record.id)
                    : registry.registerImportedModel(record.classifierClass, record.options, payload.classifier(),
                            payload.trainingHeader(), record.preprocessing, trainingData, record.id);
            ObjectNode result = modelSummary(model);
            result.put("experiment_id", record.id);
            return result;
        });
    }

    public CallToolResult saveModel(Map<String, Object> arguments) {
        return handle("weka_save_model", arguments, () -> {
            ModelEntry model = registry.model(ToolSupport.requiredString(arguments, "model_id"));
            ExperimentRecord saved = persistModel(model);
            ObjectNode result = modelSummary(model);
            result.put("experiment_id", saved.id);
            result.put("persisted", true);
            return result;
        });
    }

    public CallToolResult saveClusterer(Map<String, Object> arguments) {
        return handle("weka_save_clusterer", arguments, () -> {
            ClustererEntry entry = registry.clusterer(ToolSupport.requiredString(arguments, "clusterer_id"));
            ExperimentRecord saved = persistClusterer(entry);
            ObjectNode result = clustererSummary(entry);
            result.put("experiment_id", saved.id);
            result.put("persisted", true);
            return result;
        });
    }

    public CallToolResult loadClusterer(Map<String, Object> arguments) {
        return handle("weka_load_clusterer", arguments, () -> {
            String id = ToolSupport.requiredString(arguments, "experiment_id");
            ExperimentRecord record = experiments.get(id);
            if (!"CLUSTERER".equals(record.kind)) {
                throw new IllegalArgumentException("Experiment does not contain a clusterer: " + id);
            }
            whitelist.get(Whitelist.Kind.CLUSTERER, record.clustererClass);
            ModelCodec.ModelPayload payload = experiments.loadModel(id);
            ClustererEntry clusterer = registry.registerClusterer(record.clustererClass, record.options,
                    payload.clusterer(), payload.trainingHeader(), experiments.loadDataset(id),
                    record.assignments == null ? new int[0] : record.assignments, record.removedClassIndex, record.id);
            return clustererSummary(clusterer);
        });
    }

    public CallToolResult compareExperiments(Map<String, Object> arguments) {
        return handle("weka_compare_experiments", arguments, () -> {
            Object value = arguments.get("experiment_ids");
            if (!(value instanceof List<?> ids) || ids.size() < 2 || ids.size() > 20) {
                throw new IllegalArgumentException("experiment_ids must contain between 2 and 20 UUIDs");
            }
            String metric = ToolSupport.optionalString(arguments, "metric", "test_score");
            if (!Set.of("test_score", "validation_score").contains(metric)) {
                throw new IllegalArgumentException("metric must be test_score or validation_score");
            }
            List<ExperimentRecord> records = new ArrayList<>();
            for (Object id : ids) {
                if (!(id instanceof String experimentId)) {
                    throw new IllegalArgumentException("experiment_ids must contain only strings");
                }
                records.add(experiments.get(experimentId));
            }
            records.sort((left, right) -> {
                double leftScore = "test_score".equals(metric) ? left.testScore : left.validationScore;
                double rightScore = "test_score".equals(metric) ? right.testScore : right.validationScore;
                if (Double.isNaN(leftScore) || Double.isNaN(rightScore)) {
                    return Double.isNaN(leftScore) ? 1 : -1;
                }
                boolean rmse = "rmse".equals(records.get(0).optimize);
                return rmse ? Double.compare(leftScore, rightScore) : Double.compare(rightScore, leftScore);
            });
            ArrayNode result = array();
            for (ExperimentRecord record : records) {
                ObjectNode item = result.addObject();
                item.put("experiment_id", record.id);
                item.put("classifier_class", record.classifierClass);
                item.put("protocol", record.protocol);
                item.put("optimize", record.optimize);
                ToolSupport.putNumber(item, "validation_score", record.validationScore);
                ToolSupport.putNumber(item, "test_score", record.testScore);
                item.set("preprocessing", ToolSupport.JSON.valueToTree(record.preprocessing));
            }
            ObjectNode response = object();
            response.put("metric", metric);
            response.set("experiments", result);
            return response;
        });
    }

    public CallToolResult deleteExperiment(Map<String, Object> arguments) {
        return handle("weka_delete_experiment", arguments, () -> {
            String id = ToolSupport.requiredString(arguments, "experiment_id");
            experiments.delete(id);
            ObjectNode result = object();
            result.put("experiment_id", id);
            result.put("deleted", true);
            return result;
        });
    }

    public static List<String> defaultCandidates(Instances data) {
        if (data.classAttribute().isNominal()) {
            return List.of(
                    "weka.classifiers.trees.RandomForest",
                    "weka.classifiers.bayes.NaiveBayes",
                    "weka.classifiers.trees.J48",
                    "weka.classifiers.functions.Logistic",
                    "weka.classifiers.lazy.IBk");
        }
        return List.of(
                "weka.classifiers.functions.LinearRegression",
                "weka.classifiers.functions.SMOreg",
                "weka.classifiers.trees.RandomForest");
    }

    public static int automaticFoldCount(Instances data) {
        int folds = Math.min(10, data.numInstances());
        if (data.classAttribute().isNominal()) {
            int[] counts = new int[data.classAttribute().numValues()];
            for (Instance instance : data) {
                if (!instance.classIsMissing()) {
                    counts[(int) instance.classValue()]++;
                }
            }
            for (int count : counts) {
                if (count > 0) {
                    folds = Math.min(folds, Math.max(2, count));
                }
            }
        }
        return Math.max(2, folds);
    }

    private static JsonNode payloadOf(CallToolResult result) {
        String text = ((McpSchema.TextContent) result.content().get(0)).text();
        if (Boolean.TRUE.equals(result.isError())) {
            throw new IllegalArgumentException(text);
        }
        return ToolSupport.JSON.readTree(text);
    }

    public CallToolResult evaluateModel(Map<String, Object> arguments) {
        return handle("weka_evaluate_model", arguments, () -> {
            ModelEntry model = registry.model(ToolSupport.requiredString(arguments, "model_id"));
            DatasetEntry testEntry = registry.dataset(ToolSupport.requiredString(arguments, "test_dataset_id"));
            Instances test = new Instances(testEntry.data());
            test.setClassIndex(model.trainingHeader().classIndex());
            if (!model.trainingHeader().equalHeaders(test)) {
                throw new IllegalArgumentException("Test dataset header is incompatible with the training header");
            }
            ObjectNode result = object();
            result.put("model_id", model.id());
            result.put("dataset_id", testEntry.id());
            result.put("classifier_class", model.classifierClass());
            result.put("independent_test", true);
            result.put("instances", test.numInstances());
            result.put("class_type", test.classAttribute().isNominal() ? "nominal" : "numeric");
            result.set("preprocessing", ToolSupport.JSON.valueToTree(model.preprocessing()));
            tasks.call(config.trainTimeoutSeconds(), () -> {
                Evaluation evaluation = new Evaluation(test);
                synchronized (model) {
                    evaluation.evaluateModel(model.classifier(), test);
                }
                addEvaluationMetrics(result, evaluation, test);
                return null;
            });
            return result;
        });
    }

    public CallToolResult listResources(Map<String, Object> arguments) {
        return handle("weka_list_resources", arguments, () -> {
            ObjectNode result = object();
            ArrayNode datasets = result.putArray("datasets");
            for (DatasetEntry entry : registry.datasets()) {
                datasets.add(datasetSummary(entry));
            }
            ArrayNode models = result.putArray("models");
            for (ModelEntry entry : registry.models()) {
                models.add(modelSummary(entry));
            }
            ArrayNode clusterers = result.putArray("clusterers");
            for (ClustererEntry entry : registry.clusterers()) {
                ObjectNode item = clusterers.addObject();
                item.put("clusterer_id", entry.id());
                item.put("clusterer_class", entry.clustererClass());
                item.put("instances", entry.assignments().length);
                item.put("removed_class_index", entry.removedClassIndex() < 0 ? -1 : entry.removedClassIndex() + 1);
            }
            result.put("dataset_count", datasets.size());
            result.put("model_count", models.size());
            result.put("clusterer_count", clusterers.size());
            return result;
        });
    }

    public CallToolResult predict(Map<String, Object> arguments) {
        return handle("weka_predict", arguments, () -> {
            ModelEntry model = registry.model(ToolSupport.requiredString(arguments, "model_id"));
            String datasetId = ToolSupport.optionalString(arguments, "dataset_id", null);
            String instanceJson = ToolSupport.optionalString(arguments, "instance_json", null);
            if ((datasetId == null) == (instanceJson == null)) {
                throw new IllegalArgumentException("Specify exactly one of dataset_id or instance_json");
            }
            boolean includeDistribution = ToolSupport.optionalBoolean(arguments, "include_distribution", datasetId == null);
            Instances input;
            if (datasetId != null) {
                DatasetEntry entry = registry.dataset(datasetId);
                if (entry.data().numInstances() > config.maxPredictionRows()) {
                    throw new IllegalArgumentException("prediction rows exceed maxPredictionRows=" + config.maxPredictionRows());
                }
                input = new Instances(entry.data());
                input.setClassIndex(model.trainingHeader().classIndex());
                if (!model.trainingHeader().equalHeaders(input)) {
                    throw new IllegalArgumentException("Prediction dataset header is incompatible with the training header");
                }
            } else {
                if (instanceJson.getBytes(StandardCharsets.UTF_8).length > config.maxInstanceJsonBytes()) {
                    throw new IllegalArgumentException("instance_json exceeds maxInstanceJsonBytes=" + config.maxInstanceJsonBytes());
                }
                JsonNode json = ToolSupport.JSON.readTree(instanceJson);
                input = new Instances(model.trainingHeader(), 0);
                input.add(instanceFromJson(json, input));
            }
            Instances predictionInput = input;
            return tasks.call(config.loadTimeoutSeconds(), () -> {
                ObjectNode result = object();
                result.put("model_id", model.id());
                result.put("count", predictionInput.numInstances());
                ArrayNode predictions = result.putArray("predictions");
                synchronized (model) {
                    for (int index = 0; index < predictionInput.numInstances(); index++) {
                        predictions.add(prediction(model.classifier(), predictionInput, index, includeDistribution));
                    }
                }
                return result;
            });
        });
    }

    public CallToolResult cluster(Map<String, Object> arguments) {
        return handle("weka_cluster", arguments, () -> {
            DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            if (entry.data().numInstances() > config.maxPredictionRows()) {
                throw new IllegalArgumentException("cluster assignments exceed maxPredictionRows=" + config.maxPredictionRows());
            }
            String className = ToolSupport.requiredString(arguments, "clusterer_class");
            String[] options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
            Clusterer clusterer = (Clusterer) whitelist.create(Whitelist.Kind.CLUSTERER, className, options);
            int requestedClusters = ToolSupport.optionalInt(arguments, "num_clusters", 0);
            if (requestedClusters < 0 || requestedClusters > 1000) {
                throw new IllegalArgumentException("num_clusters must be between 1 and 1000");
            }
            if (requestedClusters > 0) {
                if (!(clusterer instanceof NumberOfClustersRequestable requestable)) {
                    throw new IllegalArgumentException(className + " does not accept num_clusters");
                }
                requestable.setNumClusters(requestedClusters);
            }
            Instances data = new Instances(entry.data());
            boolean removeClass = ToolSupport.optionalBoolean(arguments, "remove_class", true);
            int removedClassIndex = removeClass ? data.classIndex() : -1;
            if (removedClassIndex >= 0) {
                Filter remove = (Filter) whitelist.create(Whitelist.Kind.FILTER,
                        "weka.filters.unsupervised.attribute.Remove",
                        new String[]{"-R", Integer.toString(removedClassIndex + 1)});
                remove.setInputFormat(data);
                data = Filter.useFilter(data, remove);
            }
            data.setClassIndex(-1);
            Instances clusteringData = data;
            int removed = removedClassIndex;
            ClusteringResult clustering = tasks.call(config.trainTimeoutSeconds(), () -> {
                clusterer.buildClusterer(clusteringData);
                int[] assignments = new int[clusteringData.numInstances()];
                for (int index = 0; index < clusteringData.numInstances(); index++) {
                    assignments[index] = clusterer.clusterInstance(clusteringData.instance(index));
                }
                Double logLikelihood = null;
                if (clusterer instanceof weka.clusterers.DensityBasedClusterer) {
                    weka.clusterers.ClusterEvaluation evaluation = new weka.clusterers.ClusterEvaluation();
                    evaluation.setClusterer(clusterer);
                    evaluation.evaluateClusterer(clusteringData);
                    logLikelihood = evaluation.getLogLikelihood();
                }
                return new ClusteringResult(clusterer.numberOfClusters(), assignments, logLikelihood);
            });
            ObjectNode result = object();
            result.put("clusterer_id", OUTPUT_PLACEHOLDER_ID);
            result.put("clusterer_class", className);
            result.put("num_clusters", clustering.numClusters());
            result.put("instances", clusteringData.numInstances());
            result.put("attributes", clusteringData.numAttributes());
            result.put("removed_class_index", removed < 0 ? -1 : removed + 1);
            if (clustering.logLikelihood() != null) {
                ToolSupport.putNumber(result, "log_likelihood", clustering.logLikelihood());
            }
            ArrayNode assignmentNodes = result.putArray("assignments");
            for (int assignment : clustering.assignments()) {
                assignmentNodes.add(assignment);
            }
            ensureOutputWithinLimit(result);
            ClustererEntry trained = new ClustererEntry(OUTPUT_PLACEHOLDER_ID, className, options, clusterer,
                    clusteringData, clustering.assignments(), removed);
            ClustererEntry registered;
            if (ToolSupport.optionalBoolean(arguments, "persist", false)) {
                ExperimentRecord saved = persistClusterer(trained);
                registered = registry.registerClusterer(className, options, clusterer, clusteringData,
                        clusteringData, clustering.assignments(), removed, saved.id);
                result.put("experiment_id", saved.id);
            } else {
                registered = registry.registerClusterer(className, options, clusterer, clusteringData,
                        clustering.assignments(), removed);
            }
            result.put("clusterer_id", registered.id());
            return result;
        });
    }

    public CallToolResult clusterEvaluate(Map<String, Object> arguments) {
        return handle("weka_cluster_evaluate", arguments, () -> {
            ClustererEntry entry = registry.clusterer(ToolSupport.requiredString(arguments, "clusterer_id"));
            DatasetEntry dataset = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
            int maximumSamples = ToolSupport.optionalInt(arguments, "max_samples", Math.min(500, config.maxPredictionRows()));
            if (maximumSamples < 1 || maximumSamples > config.maxPredictionRows()) {
                throw new IllegalArgumentException("max_samples must be between 1 and maxPredictionRows=" + config.maxPredictionRows());
            }
            Instances aligned = alignClusterInput(entry, new Instances(dataset.data()));
            Instances raw = new Instances(dataset.data());
            boolean hasLabels = entry.removedClassIndex() >= 0 && raw.classIndex() >= 0
                    && raw.classAttribute().isNominal();
            ClusterQuality quality = tasks.call(config.trainTimeoutSeconds(), () -> {
                synchronized (entry) {
                    weka.clusterers.ClusterEvaluation evaluation = new weka.clusterers.ClusterEvaluation();
                    evaluation.setClusterer(entry.clusterer());
                    evaluation.evaluateClusterer(hasLabels ? raw : aligned);
                    int[] counts = new int[entry.clusterer().numberOfClusters()];
                    int[] assignments = new int[aligned.numInstances()];
                    for (int index = 0; index < aligned.numInstances(); index++) {
                        assignments[index] = entry.clusterer().clusterInstance(aligned.instance(index));
                        counts[assignments[index]]++;
                    }
                    return new ClusterQuality(evaluation.getLogLikelihood(), evaluation.getClassesToClusters(),
                            counts, silhouette(aligned, assignments, maximumSamples));
                }
            });
            ObjectNode result = object();
            result.put("clusterer_id", entry.id());
            result.put("dataset_id", dataset.id());
            result.put("instances", aligned.numInstances());
            result.put("num_clusters", quality.counts.length);
            result.set("cluster_sizes", ToolSupport.JSON.valueToTree(quality.counts));
            if (entry.clusterer() instanceof weka.clusterers.DensityBasedClusterer) {
                ToolSupport.putNumber(result, "log_likelihood", quality.logLikelihood);
            }
            ArrayNode classMappings = result.putArray("class_to_cluster");
            if (hasLabels && quality.classesToClusters != null) {
                for (int index = 0; index < quality.classesToClusters.length; index++) {
                    ObjectNode mapping = classMappings.addObject();
                    mapping.put("class", raw.classAttribute().value(index));
                    mapping.put("cluster", quality.classesToClusters[index]);
                }
            }
            ObjectNode silhouette = result.putObject("silhouette");
            ToolSupport.putNumber(silhouette, "mean", quality.silhouette.mean);
            silhouette.put("samples", quality.silhouette.samples);
            silhouette.put("sampled", aligned.numInstances() > quality.silhouette.samples);
            ArrayNode perCluster = silhouette.putArray("per_cluster");
            for (double score : quality.silhouette.perCluster) {
                perCluster.add(score);
            }
            return result;
        });
    }

    private static Silhouette silhouette(Instances data, int[] assignments, int maximumSamples) {
        int stride = Math.max(1, (data.numInstances() + maximumSamples - 1) / maximumSamples);
        List<Integer> samples = new ArrayList<>();
        for (int index = 0; index < data.numInstances() && samples.size() < maximumSamples; index += stride) {
            samples.add(index);
        }
        weka.core.EuclideanDistance distance = new weka.core.EuclideanDistance();
        distance.setInstances(data);
        double[] perCluster = new double[data.numInstances() == 0 ? 0 : maximums(assignments)];
        int[] clusterCounts = new int[perCluster.length];
        double total = 0;
        for (int sample : samples) {
            double[] sums = new double[perCluster.length];
            int[] counts = new int[perCluster.length];
            for (int other : samples) {
                if (sample == other) {
                    continue;
                }
                double value = distance.distance(data.instance(sample), data.instance(other));
                if (Double.isFinite(value)) {
                    sums[assignments[other]] += value;
                    counts[assignments[other]]++;
                }
            }
            int own = assignments[sample];
            double a = counts[own] > 0 ? sums[own] / counts[own] : 0;
            double b = Double.POSITIVE_INFINITY;
            for (int cluster = 0; cluster < sums.length; cluster++) {
                if (cluster != own && counts[cluster] > 0) {
                    b = Math.min(b, sums[cluster] / counts[cluster]);
                }
            }
            double score = counts[own] == 0 || !Double.isFinite(b) || Math.max(a, b) == 0
                    ? 0 : (b - a) / Math.max(a, b);
            perCluster[own] += score;
            clusterCounts[own]++;
            total += score;
        }
        for (int index = 0; index < perCluster.length; index++) {
            perCluster[index] = clusterCounts[index] == 0 ? Double.NaN : perCluster[index] / clusterCounts[index];
        }
        return new Silhouette(samples.isEmpty() ? Double.NaN : total / samples.size(), perCluster, samples.size());
    }

    private static int maximums(int[] values) {
        int maximum = 0;
        for (int value : values) {
            maximum = Math.max(maximum, value + 1);
        }
        return maximum;
    }

    private record Silhouette(double mean, double[] perCluster, int samples) {
    }

    private record ClusterQuality(double logLikelihood, int[] classesToClusters, int[] counts, Silhouette silhouette) {
    }

    public CallToolResult clusterPredict(Map<String, Object> arguments) {
        return handle("weka_cluster_predict", arguments, () -> {
            ClustererEntry entry = registry.clusterer(ToolSupport.requiredString(arguments, "clusterer_id"));
            String datasetId = ToolSupport.optionalString(arguments, "dataset_id", null);
            String instanceJson = ToolSupport.optionalString(arguments, "instance_json", null);
            if ((datasetId == null) == (instanceJson == null)) {
                throw new IllegalArgumentException("Specify exactly one of dataset_id or instance_json");
            }
            boolean includeDistribution = ToolSupport.optionalBoolean(arguments, "include_distribution", false);
            if (includeDistribution && !(entry.clusterer() instanceof weka.clusterers.DensityBasedClusterer)) {
                throw new IllegalArgumentException("Cluster distributions require a density-based clusterer");
            }
            Instances input;
            if (datasetId != null) {
                DatasetEntry dataset = registry.dataset(datasetId);
                if (dataset.data().numInstances() > config.maxPredictionRows()) {
                    throw new IllegalArgumentException("prediction rows exceed maxPredictionRows=" + config.maxPredictionRows());
                }
                input = alignClusterInput(entry, new Instances(dataset.data()));
            } else {
                if (instanceJson.getBytes(StandardCharsets.UTF_8).length > config.maxInstanceJsonBytes()) {
                    throw new IllegalArgumentException("instance_json exceeds maxInstanceJsonBytes=" + config.maxInstanceJsonBytes());
                }
                input = new Instances(entry.trainingHeader(), 0);
                input.add(instanceFromJson(ToolSupport.JSON.readTree(instanceJson), input));
            }
            Instances predictionInput = input;
            return tasks.call(config.loadTimeoutSeconds(), () -> {
                ObjectNode result = object();
                result.put("clusterer_id", entry.id());
                result.put("count", predictionInput.numInstances());
                ArrayNode predictions = result.putArray("predictions");
                synchronized (entry) {
                    for (int index = 0; index < predictionInput.numInstances(); index++) {
                        Instance instance = predictionInput.instance(index);
                        ObjectNode item = predictions.addObject();
                        item.put("index", index);
                        item.put("cluster", entry.clusterer().clusterInstance(instance));
                        if (includeDistribution) {
                            ArrayNode distribution = item.putArray("cluster_distribution");
                            for (double value : entry.clusterer().distributionForInstance(instance)) {
                                distribution.add(value);
                            }
                        }
                    }
                }
                return result;
            });
        });
    }

    private Instances alignClusterInput(ClustererEntry entry, Instances input) throws Exception {
        Instances aligned = input;
        if (entry.removedClassIndex() >= 0
                && aligned.numAttributes() == entry.trainingHeader().numAttributes() + 1) {
            Filter remove = (Filter) whitelist.create(Whitelist.Kind.FILTER,
                    "weka.filters.unsupervised.attribute.Remove",
                    new String[]{"-R", Integer.toString(entry.removedClassIndex() + 1)});
            remove.setInputFormat(aligned);
            aligned = Filter.useFilter(aligned, remove);
        }
        aligned.setClassIndex(-1);
        if (!entry.trainingHeader().equalHeaders(aligned)) {
            throw new IllegalArgumentException("Prediction dataset header is incompatible with the clustering header");
        }
        return aligned;
    }

    public CallToolResult release(Map<String, Object> arguments) {
        return handle("weka_release", arguments, () -> {
            String id = ToolSupport.requiredString(arguments, "resource_id");
            String type = registry.release(id);
            ObjectNode result = object();
            result.put("resource_id", id);
            result.put("resource_type", type);
            result.put("released", true);
            return result;
        });
    }

    private SyncToolSpecification specification(
            String name, String description, Map<String, Object> schema,
            Function<Map<String, Object>, CallToolResult> handler) {
        return SyncToolSpecification.builder()
                .tool(McpSchema.Tool.builder(name, schema).description(description).build())
                .callHandler((exchange, request) -> handler.apply(
                        request.arguments() == null ? Map.of() : request.arguments()))
                .build();
    }

    private CallToolResult handle(String tool, Map<String, Object> arguments, ThrowingSupplier<JsonNode> action) {
        long started = System.nanoTime();
        Map<String, Object> audit = new LinkedHashMap<>();
        AUDIT_KEYS.forEach(key -> {
            if (arguments.containsKey(key)) {
                audit.put(key, arguments.get(key));
            }
        });
        try {
            JsonNode payload = action.get();
            ensureOutputWithinLimit(payload);
            String json = ToolSupport.JSON.writeValueAsString(payload);
            LOGGER.info("tool={} arguments={} elapsed_ms={} status=OK", tool, audit, elapsed(started));
            return ToolSupport.success(json);
        } catch (Exception e) {
            String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            LOGGER.warn("tool={} arguments={} elapsed_ms={} status=ERROR error={}", tool, audit, elapsed(started), message);
            return ToolSupport.error(message);
        }
    }

    private void ensureOutputWithinLimit(JsonNode payload) {
        String json = ToolSupport.JSON.writeValueAsString(payload);
        if (json.getBytes(StandardCharsets.UTF_8).length > config.maxOutputBytes()) {
            throw new IllegalStateException("tool output exceeds maxOutputBytes=" + config.maxOutputBytes());
        }
    }

    private static long elapsed(long started) {
        return (System.nanoTime() - started) / 1_000_000;
    }

    private void validateDataset(Instances data) {
        if (data.numAttributes() < 1 || data.numAttributes() > config.maxAttributes()) {
            throw new IllegalArgumentException("attribute count exceeds maxAttributes=" + config.maxAttributes());
        }
        if (data.numInstances() > config.maxInstances()) {
            throw new IllegalArgumentException("instance count exceeds maxInstances=" + config.maxInstances());
        }
    }

    public static Instances classificationData(DatasetEntry entry, int requestedClassIndex) {
        Instances data = new Instances(entry.data());
        setRequestedClassIndex(data, requestedClassIndex);
        if (data.classIndex() < 0) {
            if (data.numAttributes() == 0) {
                throw new IllegalArgumentException("Dataset has no class attribute");
            }
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    public static DataSplit splitData(Instances data, double trainPercent, long seed) {
        Instances randomized = new Instances(data);
        randomized.randomize(new Random(seed));
        Instances training = new Instances(randomized, 0);
        Instances test = new Instances(randomized, 0);
        if (data.classAttribute().isNominal()) {
            Map<Integer, List<Instance>> groups = new LinkedHashMap<>();
            for (Instance instance : randomized) {
                int group = instance.classIsMissing() ? -1 : (int) instance.classValue();
                groups.computeIfAbsent(group, ignored -> new ArrayList<>()).add(instance);
            }
            long groupSeed = seed;
            for (List<Instance> group : groups.values()) {
                java.util.Collections.shuffle(group, new Random(groupSeed++));
                int trainSize = (int) Math.round(group.size() * trainPercent / 100.0);
                if (group.size() > 1) {
                    trainSize = Math.max(1, Math.min(group.size() - 1, trainSize));
                } else {
                    trainSize = trainPercent >= 50 ? 1 : 0;
                }
                for (int index = 0; index < group.size(); index++) {
                    (index < trainSize ? training : test).add(group.get(index));
                }
            }
            training.randomize(new Random(seed));
            test.randomize(new Random(seed + 1));
        } else {
            int trainSize = (int) Math.round(randomized.numInstances() * trainPercent / 100.0);
            for (int index = 0; index < randomized.numInstances(); index++) {
                (index < trainSize ? training : test).add(randomized.instance(index));
            }
        }
        training.setClassIndex(data.classIndex());
        test.setClassIndex(data.classIndex());
        if (training.numInstances() == 0 || test.numInstances() == 0) {
            throw new IllegalArgumentException("train_percent produces an empty training or test set");
        }
        return new DataSplit(training, test);
    }

    private static void setRequestedClassIndex(Instances data, int requestedClassIndex) {
        if (requestedClassIndex < 0 || requestedClassIndex > data.numAttributes()) {
            throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
        }
        if (requestedClassIndex > 0) {
            data.setClassIndex(requestedClassIndex - 1);
        }
    }

    private static ObjectNode datasetSummary(DatasetEntry entry) {
        ObjectNode result = object();
        result.put("dataset_id", entry.id());
        result.put("name", entry.name());
        result.put("instances", entry.data().numInstances());
        result.put("attributes", entry.data().numAttributes());
        result.put("class_index", entry.data().classIndex() < 0 ? -1 : entry.data().classIndex() + 1);
        return result;
    }

    private ExperimentRecord persistModel(ModelEntry model) throws Exception {
        ExperimentRecord previous = model.experimentId() == null ? null : experiments.get(model.experimentId());
        ExperimentRecord record = new ExperimentRecord();
        record.kind = "CLASSIFIER";
        record.classifierClass = model.classifierClass();
        record.options = model.options();
        record.preprocessing = model.preprocessing();
        record.classIndex = model.trainingHeader().classIndex();
        record.attributes = model.trainingHeader().numAttributes();
        record.retrainable = model.retrainable();
        Instances training = model.trainingData();
        if (previous != null) {
            record.parentExperimentId = previous.id;
            record.datasetRole = previous.datasetRole;
            record.interoperability = previous.interoperability;
            record.candidates = previous.candidates;
            record.search = previous.search;
            record.sourcePath = previous.sourcePath;
            record.protocol = previous.protocol;
            record.stream = previous.stream;
            record.finalModelTrainedOnFullData = previous.finalModelTrainedOnFullData;
            record.validationMetrics = previous.validationMetrics;
            record.testMetrics = previous.testMetrics;
            record.validationScore = previous.validationScore;
            record.testScore = previous.testScore;
            record.optimize = previous.optimize;
            record.selectionMetric = previous.selectionMetric;
            record.seed = previous.seed;
            record.instances = previous.instances;
            if (training == null && record.sourcePath == null) {
                training = experiments.loadDataset(previous.id);
            }
        }
        if (training == null && record.sourcePath == null) {
            throw new IllegalArgumentException("Model has no training data or source path");
        }
        if (training != null) {
            record.instances = training.numInstances();
        }
        if (record.protocol == null) {
            record.protocol = "training";
        }
        return experiments.create(record, training, model.classifier(), model.trainingHeader());
    }

    private ExperimentRecord persistClusterer(ClustererEntry clusterer) throws Exception {
        if (clusterer.trainingData() == null) {
            throw new IllegalArgumentException("Clusterer has no training data");
        }
        ExperimentRecord record = new ExperimentRecord();
        record.kind = "CLUSTERER";
        record.clustererClass = clusterer.clustererClass();
        record.options = clusterer.options();
        record.classIndex = -1;
        record.instances = clusterer.trainingData().numInstances();
        record.attributes = clusterer.trainingHeader().numAttributes();
        record.protocol = "clustering";
        record.removedClassIndex = clusterer.removedClassIndex();
        record.assignments = clusterer.assignments();
        return experiments.create(record, clusterer.trainingData(), clusterer.clusterer(), clusterer.trainingHeader());
    }

    private static ObjectNode clustererSummary(ClustererEntry clusterer) throws Exception {
        ObjectNode result = object();
        result.put("clusterer_id", clusterer.id());
        result.put("clusterer_class", clusterer.clustererClass());
        result.put("num_clusters", clusterer.clusterer().numberOfClusters());
        result.put("instances", clusterer.assignments().length);
        result.put("attributes", clusterer.trainingHeader().numAttributes());
        result.put("removed_class_index", clusterer.removedClassIndex() < 0 ? -1 : clusterer.removedClassIndex() + 1);
        result.set("options", ToolSupport.JSON.valueToTree(clusterer.options()));
        if (clusterer.experimentId() != null) {
            result.put("experiment_id", clusterer.experimentId());
        }
        return result;
    }

    public static ObjectNode modelSummary(ModelEntry model) {
        ObjectNode result = object();
        result.put("model_id", model.id());
        result.put("classifier_class", model.classifierClass());
        result.put("class_index", model.trainingHeader().classIndex() + 1);
        result.put("class_attribute", model.trainingHeader().classAttribute().name());
        result.put("class_type", model.trainingHeader().classAttribute().isNominal() ? "nominal" : "numeric");
        result.set("class_labels", ToolSupport.JSON.valueToTree(ToolSupport.classLabels(model.trainingHeader())));
        result.set("options", ToolSupport.JSON.valueToTree(model.options()));
        result.set("preprocessing", ToolSupport.JSON.valueToTree(model.preprocessing()));
        result.put("retrainable", model.retrainable());
        if (model.experimentId() != null) {
            result.put("experiment_id", model.experimentId());
        }
        return result;
    }

    public static void addEvaluationMetrics(ObjectNode result, Evaluation evaluation, Instances data) throws Exception {
        if (data.classAttribute().isNominal()) {
            ToolSupport.putNumber(result, "accuracy_percent", evaluation.pctCorrect());
            ToolSupport.putNumber(result, "kappa", evaluation.kappa());
            result.put("correct", evaluation.correct());
            result.put("incorrect", evaluation.incorrect());
            result.put("unclassified", evaluation.unclassified());
            ArrayNode matrix = result.putArray("confusion_matrix");
            for (double[] row : evaluation.confusionMatrix()) {
                ArrayNode rowNode = matrix.addArray();
                for (double value : row) {
                    rowNode.add(value);
                }
            }
            result.set("class_labels", ToolSupport.JSON.valueToTree(ToolSupport.classLabels(data)));
            ArrayNode perClass = result.putArray("per_class");
            for (int index = 0; index < data.classAttribute().numValues(); index++) {
                ObjectNode item = perClass.addObject();
                item.put("class_label", data.classAttribute().value(index));
                ToolSupport.putNumber(item, "support", evaluation.numTruePositives(index)
                        + evaluation.numFalseNegatives(index));
                ToolSupport.putNumber(item, "precision", evaluation.precision(index));
                ToolSupport.putNumber(item, "recall", evaluation.recall(index));
                ToolSupport.putNumber(item, "f1", evaluation.fMeasure(index));
                ToolSupport.putNumber(item, "auc", evaluation.areaUnderROC(index));
                ToolSupport.putNumber(item, "false_positive_rate", evaluation.falsePositiveRate(index));
            }
        } else {
            ToolSupport.putNumber(result, "mae", evaluation.meanAbsoluteError());
            ToolSupport.putNumber(result, "rmse", evaluation.rootMeanSquaredError());
            ToolSupport.putNumber(result, "correlation_coefficient", evaluation.correlationCoefficient());
            ToolSupport.putNumber(result, "rae_percent", evaluation.relativeAbsoluteError());
            ToolSupport.putNumber(result, "rrse_percent", evaluation.rootRelativeSquaredError());
        }
    }

    private static ObjectNode prediction(
            Classifier classifier, Instances data, int index, boolean includeDistribution) throws Exception {
        Instance instance = data.instance(index);
        ObjectNode result = object();
        result.put("index", index);
        if (data.classAttribute().isNominal()) {
            double[] distribution = classifier.distributionForInstance(instance);
            int predicted = weka.core.Utils.maxIndex(distribution);
            result.put("predicted_value", data.classAttribute().value(predicted));
            if (includeDistribution) {
                ObjectNode distributionNode = result.putObject("class_distribution");
                for (int value = 0; value < distribution.length; value++) {
                    ToolSupport.putNumber(distributionNode, data.classAttribute().value(value), distribution[value]);
                }
            }
        } else {
            ToolSupport.putNumber(result, "predicted_value", classifier.classifyInstance(instance));
        }
        return result;
    }

    private static Instance instanceFromJson(JsonNode json, Instances header) {
        DenseInstance instance = new DenseInstance(header.numAttributes());
        instance.setDataset(header);
        if (json == null || (!json.isObject() && !json.isArray())) {
            throw new IllegalArgumentException("instance_json must be a JSON object or array");
        }
        if (json.isObject()) {
            for (Map.Entry<String, JsonNode> field : json.properties()) {
                Attribute attribute = header.attribute(field.getKey());
                if (attribute == null) {
                    throw new IllegalArgumentException("Unknown attribute in instance_json: " + field.getKey());
                }
                setInstanceValue(instance, attribute, field.getValue());
            }
        } else {
            int classIndex = header.classIndex();
            if (json.size() != header.numAttributes()
                    && !(classIndex >= 0 && json.size() == header.numAttributes() - 1)) {
                throw new IllegalArgumentException("instance_json array must include all attributes, or all except the class");
            }
            int inputIndex = 0;
            for (int index = 0; index < header.numAttributes(); index++) {
                if (json.size() == header.numAttributes() - 1 && index == classIndex) {
                    continue;
                }
                setInstanceValue(instance, header.attribute(index), json.get(inputIndex++));
            }
        }
        return instance;
    }

    private static void setInstanceValue(Instance instance, Attribute attribute, JsonNode value) {
        try {
            if (value == null || value.isNull() || (value.isTextual() && "?".equals(value.asText()))) {
                instance.setMissing(attribute);
            } else if (attribute.isNumeric()) {
                double numeric = value.isNumber() ? value.doubleValue() : Double.parseDouble(value.asText());
                if (!Double.isFinite(numeric)) {
                    throw new IllegalArgumentException("Numeric value for " + attribute.name() + " must be finite");
                }
                instance.setValue(attribute, numeric);
            } else if (attribute.isNominal()) {
                String text = value.asText();
                if (attribute.indexOfValue(text) < 0) {
                    throw new IllegalArgumentException("Invalid nominal value for " + attribute.name() + ": " + text);
                }
                instance.setValue(attribute, text);
            } else if (attribute.isString()) {
                instance.setValue(attribute, value.asText());
            } else if (attribute.isDate()) {
                instance.setValue(attribute, attribute.parseDate(value.asText()));
            } else {
                throw new IllegalArgumentException("Relational attributes are not supported in instance_json");
            }
        } catch (Exception e) {
            if (e instanceof IllegalArgumentException illegalArgument) {
                throw illegalArgument;
            }
            throw new IllegalArgumentException("Invalid value for attribute " + attribute.name(), e);
        }
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

    private static Map<String, Object> fields() {
        return new LinkedHashMap<>();
    }

    private static Map<String, Object> loadSchema() {
        Map<String, Object> fields = fields();
        fields.put("path", ToolSupport.string("Absolute path or a path relative to the configured data root"));
        fields.put("format", Map.of("type", "string", "enum", List.of("auto", "arff", "csv"), "default", "auto"));
        fields.put("name", ToolSupport.string("Optional dataset display name"));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        return ToolSupport.schema(fields, "path");
    }

    private static Map<String, Object> idSchema(String id) {
        return ToolSupport.schema(Map.of(id, ToolSupport.string("UUID returned by a previous tool call")), id);
    }

    private static Map<String, Object> listAlgorithmsSchema() {
        return ToolSupport.schema(Map.of("kind", Map.of(
                "type", "string", "enum", List.of("all", "classifier", "filter", "clusterer", "associator"),
                "default", "all")));
    }

    private static Map<String, Object> recommendSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("limit", ToolSupport.integer("Maximum recommendations", 1, 20));
        fields.put("streaming", ToolSupport.bool("Only recommend UpdateableClassifier implementations; default false"));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> splitDatasetSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("train_percent", Map.of("type", "number", "minimum", 0.01, "maximum", 99.99, "default", 80.0));
        fields.put("seed", ToolSupport.integer("Random seed", Integer.MIN_VALUE, Integer.MAX_VALUE));
        fields.put("name_prefix", ToolSupport.string("Prefix for the new session dataset names"));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> previewSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("filter_class", ToolSupport.string("Whitelisted Weka filter class"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("max_rows", ToolSupport.integer("Maximum preview rows", 0, 100_000));
        return ToolSupport.schema(fields, "dataset_id", "filter_class");
    }

    private static Map<String, Object> histogramSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("attribute_index", ToolSupport.integer("One-based numeric or nominal attribute index", 1, 10_000));
        fields.put("bins", ToolSupport.integer("Numeric histogram bin count", 1, 100));
        return ToolSupport.schema(fields, "dataset_id", "attribute_index");
    }

    private static Map<String, Object> scatterSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("x_index", ToolSupport.integer("One-based numeric x attribute", 1, 10_000));
        fields.put("y_index", ToolSupport.integer("One-based numeric y attribute", 1, 10_000));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("max_points", ToolSupport.integer("Maximum sampled points", 1, 100_000));
        return ToolSupport.schema(fields, "dataset_id", "x_index", "y_index");
    }

    private static Map<String, Object> scatterMatrixSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("attributes", Map.of(
                "type", "array", "items", Map.of("type", "integer", "minimum", 1, "maximum", 10_000),
                "minItems", 2, "maxItems", 5));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("max_points", ToolSupport.integer("Maximum sampled points per plot", 1, 100_000));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> compareDatasetsSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Reference dataset UUID"));
        fields.put("other_dataset_id", ToolSupport.string("Dataset UUID to compare"));
        fields.put("bins", ToolSupport.integer("Numeric distribution bin count", 1, 100));
        fields.put("max_attributes", ToolSupport.integer("Maximum common attributes to analyze", 1, 100));
        return ToolSupport.schema(fields, "dataset_id", "other_dataset_id");
    }

    private static Map<String, Object> exportDatasetSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("path", ToolSupport.string("Destination ARFF or CSV path inside the configured data root"));
        fields.put("format", Map.of("type", "string", "enum", List.of("arff", "csv"), "default", "arff"));
        fields.put("overwrite", ToolSupport.bool("Replace an existing regular file; default false"));
        return ToolSupport.schema(fields, "dataset_id", "path");
    }

    private static Map<String, Object> filterSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("filter_class", ToolSupport.string("Whitelisted Weka filter class"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        return ToolSupport.schema(fields, "dataset_id", "filter_class");
    }

    private static Map<String, Object> learningCurveSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("classifier_class", ToolSupport.string("Whitelisted classifier class; default RandomForest"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("fractions", ToolSupport.string("Comma-separated training fractions in (0, 1]"));
        fields.put("seed", ToolSupport.integer("Random seed", Integer.MIN_VALUE, Integer.MAX_VALUE));
        fields.put("auto_preprocess", ToolSupport.bool("Fit preprocessing on each training subset; default true"));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> modelDiagnosticsSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Existing trained model UUID"));
        fields.put("dataset_id", ToolSupport.string("Labeled dataset UUID compatible with the training header"));
        fields.put("bins", ToolSupport.integer("Calibration and residual histogram bins", 1, 100));
        fields.put("max_errors", ToolSupport.integer("Maximum error instances returned", 0, 1000));
        return ToolSupport.schema(fields, "model_id", "dataset_id");
    }

    private static Map<String, Object> thresholdAnalysisSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Existing trained binary classifier UUID"));
        fields.put("dataset_id", ToolSupport.string("Labeled dataset UUID compatible with the training header"));
        fields.put("positive_class", ToolSupport.string("Positive nominal class label; default second class value"));
        fields.put("thresholds", ToolSupport.string("Optional comma-separated probabilities in [0, 1]"));
        fields.put("steps", ToolSupport.integer("Number of evenly spaced thresholds including 0 and 1", 2, 201));
        fields.put("cost_fp", Map.of("type", "number", "minimum", 0.0, "default", 1.0));
        fields.put("cost_fn", Map.of("type", "number", "minimum", 0.0, "default", 1.0));
        fields.put("bins", ToolSupport.integer("Margin histogram bins", 1, 100));
        return ToolSupport.schema(fields, "model_id", "dataset_id");
    }

    private static Map<String, Object> compareModelsSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("model_ids", Map.of(
                "type", "array", "items", Map.of("type", "string"), "minItems", 2, "maxItems", 8));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("metric", Map.of("type", "string", "enum", List.of("accuracy", "weighted_f1", "rmse")));
        fields.put("folds", ToolSupport.integer("Shared cross-validation folds", 2, 100));
        fields.put("runs", ToolSupport.integer("Repeated cross-validation runs", 1, 20));
        fields.put("seed", ToolSupport.integer("Random seed", Integer.MIN_VALUE, Integer.MAX_VALUE));
        fields.put("confidence", Map.of("type", "number", "minimum", 0.5, "exclusiveMaximum", 1.0, "default", 0.95));
        fields.put("budgetSeconds", ToolSupport.integer("Total comparison budget in seconds", 1, 86400));
        return ToolSupport.schema(fields, "dataset_id", "model_ids");
    }

    private static Map<String, Object> selectAttributesSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("method", Map.of("type", "string", "enum",
                List.of("info_gain", "cfs", "gain_ratio", "symmetrical_uncert", "one_r", "relief_f"), "default", "info_gain"));
        fields.put("max_attributes", ToolSupport.integer("Maximum selected attributes", 1, 10_000));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> associateSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("algorithm", Map.of("type", "string", "enum", List.of("apriori", "fpgrowth"), "default", "apriori"));
        fields.put("min_support", Map.of("type", "number", "minimum", 0.000001, "maximum", 1.0, "default", 0.1));
        fields.put("min_confidence", Map.of("type", "number", "minimum", 0.0, "maximum", 1.0, "default", 0.5));
        fields.put("max_rules", ToolSupport.integer("Maximum returned rules", 1, 1000));
        fields.put("auto_prepare", ToolSupport.bool("Discretize numeric data and binary-encode nominal data for FPGrowth; default true"));
        fields.put("options", ToolSupport.optionsSchema());
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> trainSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("classifier_class", ToolSupport.string("Whitelisted Weka classifier class"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("One-based class index; defaults to the dataset class or last attribute", 1, 10_000));
        fields.put("auto_preprocess", ToolSupport.bool("Build and persist a fold-safe preprocessing pipeline for this model"));
        fields.put("persist", ToolSupport.bool("Save the model and training data in a persistent experiment; default false"));
        return ToolSupport.schema(fields, "dataset_id", "classifier_class");
    }

    private static Map<String, Object> trainStreamSchema() {
        Map<String, Object> fields = fields();
        fields.put("path", ToolSupport.string("ARFF or CSV path inside the configured data root"));
        fields.put("format", Map.of("type", "string", "enum", List.of("auto", "arff", "csv"), "default", "auto"));
        fields.put("classifier_class", ToolSupport.string("Whitelisted UpdateableClassifier implementation"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("One-based class index; default last attribute", 1, 10_000));
        fields.put("evaluation", Map.of(
                "type", "string", "enum", List.of("holdout", "prequential"), "default", "holdout"));
        fields.put("test_every", ToolSupport.integer("Evaluate every Nth labeled instance", 1, 1_000_000));
        fields.put("max_instances", ToolSupport.integer("Maximum instances to read", 1, 100_000));
        fields.put("max_test_instances", ToolSupport.integer("Bounded evaluation count; zero disables evaluation", 0, 100_000));
        fields.put("budgetSeconds", ToolSupport.integer("Total incremental training budget in seconds", 1, 86400));
        fields.put("persist", ToolSupport.bool("Persist the model with a source-file reference; default true"));
        return ToolSupport.schema(fields, "path", "classifier_class");
    }

    private static Map<String, Object> autoClassifySchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("classifier_class", ToolSupport.string("Whitelisted classifier or 'auto' to compare task-appropriate candidates"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("auto_preprocess", ToolSupport.bool("Automatically fit preprocessing inside training folds; default true"));
        fields.put("evaluate", ToolSupport.bool("Evaluate before final full-data training; default true"));
        fields.put("folds", ToolSupport.integer("Inner or legacy cross-validation folds", 2, 100_000));
        fields.put("outer_folds", ToolSupport.integer("Nested CV outer folds", 2, 100_000));
        fields.put("inner_folds", ToolSupport.integer("Nested CV inner folds", 2, 100_000));
        fields.put("seed", ToolSupport.integer("Random seed", Integer.MIN_VALUE, Integer.MAX_VALUE));
        fields.put("mode", Map.of("type", "string", "enum", List.of("cross_validation", "holdout"), "default", "cross_validation"));
        fields.put("protocol", Map.of("type", "string", "enum", List.of("holdout", "nested_cv", "legacy_cv"), "default", "holdout"));
        fields.put("search", Map.of("type", "string", "enum", List.of("defaults", "grid", "random", "halving"), "default", "defaults"));
        fields.put("optimize", Map.of("type", "string", "enum", List.of("accuracy", "weighted_f1", "rmse")));
        fields.put("maxCandidates", ToolSupport.integer("Maximum candidates including the ZeroR baseline", 1, 1000));
        fields.put("candidates", Map.of(
                "type", "array", "maxItems", 1000,
                "items", Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "classifier_class", ToolSupport.string("Whitelisted classifier class"),
                                "options", ToolSupport.optionsSchema()),
                        "required", List.of("classifier_class"),
                        "additionalProperties", false)));
        fields.put("include_baseline", ToolSupport.bool("Include the ZeroR baseline when it is not already present; default true"));
        fields.put("budgetSeconds", ToolSupport.integer("Total AutoML budget in seconds", 1, 86400));
        fields.put("train_percent", Map.of("type", "number", "minimum", 0.01, "maximum", 99.99, "default", 80.0));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> listExperimentsSchema() {
        Map<String, Object> fields = fields();
        fields.put("offset", ToolSupport.integer("Pagination offset", 0, Integer.MAX_VALUE));
        fields.put("limit", ToolSupport.integer("Page size", 1, 100));
        return ToolSupport.schema(fields);
    }

    private static Map<String, Object> experimentIdSchema() {
        return ToolSupport.schema(Map.of("experiment_id", ToolSupport.string("Persistent experiment UUID")), "experiment_id");
    }

    private static Map<String, Object> exportModelSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Classifier UUID to export"));
        fields.put("clusterer_id", ToolSupport.string("Clusterer UUID to export"));
        fields.put("path", ToolSupport.string("Destination .model path inside the configured data root"));
        fields.put("format", Map.of("type", "string", "enum", List.of("weka_bundle", "raw_model", "pmml"), "default", "weka_bundle"));
        fields.put("overwrite", ToolSupport.bool("Replace an existing regular file; default false"));
        return ToolSupport.schema(fields, "path");
    }

    private static Map<String, Object> importModelSchema() {
        Map<String, Object> fields = fields();
        fields.put("path", ToolSupport.string("Trusted Weka .model file inside the configured data root"));
        fields.put("dataset_id", ToolSupport.string("Dataset used to validate or supply the training header"));
        fields.put("format", Map.of(
                "type", "string", "enum", List.of("auto", "raw_model", "weka_bundle"), "default", "auto"));
        fields.put("trusted", ToolSupport.bool("Required true only for trusted Java-serialized models"));
        fields.put("class_index", ToolSupport.integer("Optional one-based class index for a classifier", 1, 10_000));
        fields.put("remove_class", ToolSupport.bool("Remove a dataset class before validating a clusterer; default true"));
        fields.put("persist", ToolSupport.bool("Save the imported model and bound dataset; default true"));
        return ToolSupport.schema(fields, "path", "dataset_id", "trusted");
    }

    private static Map<String, Object> updateModelSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Classifier UUID implementing UpdateableClassifier"));
        fields.put("dataset_id", ToolSupport.string("Labeled dataset with a training-compatible header"));
        fields.put("path", ToolSupport.string("ARFF/CSV file to consume incrementally"));
        fields.put("format", Map.of("type", "string", "enum", List.of("auto", "arff", "csv"), "default", "auto"));
        fields.put("max_instances", ToolSupport.integer("Maximum update instances to read", 1, 100_000));
        fields.put("budgetSeconds", ToolSupport.integer("Total incremental update budget in seconds", 1, 86400));
        return ToolSupport.schema(fields, "model_id");
    }

    private static Map<String, Object> compareExperimentsSchema() {
        Map<String, Object> fields = fields();
        fields.put("experiment_ids", Map.of(
                "type", "array", "items", Map.of("type", "string"), "minItems", 2, "maxItems", 20));
        fields.put("metric", Map.of("type", "string", "enum", List.of("test_score", "validation_score"), "default", "test_score"));
        return ToolSupport.schema(fields, "experiment_ids");
    }

    private static Map<String, Object> evaluateModelSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Existing trained model UUID"));
        fields.put("test_dataset_id", ToolSupport.string("Independent labeled test dataset UUID"));
        return ToolSupport.schema(fields, "model_id", "test_dataset_id");
    }

    private static Map<String, Object> evaluateSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered labeled dataset UUID"));
        fields.put("classifier_class", ToolSupport.string("Whitelisted classifier; required when model_id is absent"));
        fields.put("model_id", ToolSupport.string("Trained model whose class and options are reused; a fresh model is evaluated"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("class_index", ToolSupport.integer("Optional one-based class attribute index", 1, 10_000));
        fields.put("folds", ToolSupport.integer("Cross-validation folds", 2, 100_000));
        fields.put("seed", ToolSupport.integer("Random seed", Integer.MIN_VALUE, Integer.MAX_VALUE));
        fields.put("mode", Map.of("type", "string", "enum", List.of("cross_validation", "holdout"), "default", "cross_validation"));
        fields.put("train_percent", Map.of("type", "number", "minimum", 0.01, "maximum", 99.99, "default", 66.7));
        fields.put("repeats", ToolSupport.integer("Number of repeated evaluations with different seeds", 1, 20));
        fields.put("auto_preprocess", ToolSupport.bool("Fit preprocessing inside each training fold or training split"));
        return ToolSupport.schema(fields, "dataset_id");
    }

    private static Map<String, Object> predictSchema() {
        Map<String, Object> fields = fields();
        fields.put("model_id", ToolSupport.string("Trained model UUID"));
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID with a training-compatible header"));
        fields.put("instance_json", ToolSupport.string("JSON object keyed by attribute name, or an ordered JSON array"));
        fields.put("include_distribution", ToolSupport.bool("Include nominal class probabilities"));
        return ToolSupport.schema(fields, "model_id");
    }

    private static Map<String, Object> clusterSchema() {
        Map<String, Object> fields = fields();
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("clusterer_class", ToolSupport.string("Whitelisted Weka clusterer class"));
        fields.put("options", ToolSupport.optionsSchema());
        fields.put("num_clusters", ToolSupport.integer("Requested cluster count when supported by the clusterer", 1, 1000));
        fields.put("remove_class", ToolSupport.bool("Remove the class attribute before clustering; default true"));
        fields.put("persist", ToolSupport.bool("Save the clusterer and training data in a persistent experiment; default false"));
        return ToolSupport.schema(fields, "dataset_id", "clusterer_class");
    }

    private static Map<String, Object> clusterEvaluateSchema() {
        Map<String, Object> fields = fields();
        fields.put("clusterer_id", ToolSupport.string("Existing clusterer UUID"));
        fields.put("dataset_id", ToolSupport.string("Dataset UUID used to evaluate the clustering"));
        fields.put("max_samples", ToolSupport.integer("Maximum instances for quadratic silhouette sampling", 1, 100_000));
        return ToolSupport.schema(fields, "clusterer_id", "dataset_id");
    }

    private static Map<String, Object> clusterPredictSchema() {
        Map<String, Object> fields = fields();
        fields.put("clusterer_id", ToolSupport.string("Existing clusterer UUID"));
        fields.put("dataset_id", ToolSupport.string("Registered dataset UUID"));
        fields.put("instance_json", ToolSupport.string("One JSON instance using the post-removal cluster attributes"));
        fields.put("include_distribution", ToolSupport.bool("Include density-based cluster probabilities when supported"));
        return ToolSupport.schema(fields, "clusterer_id");
    }

    private record ClusteringResult(int numClusters, int[] assignments, Double logLikelihood) {
    }

    public record DataSplit(Instances training, Instances test) {
    }

    @FunctionalInterface
    private interface ThrowingSupplier<T> {
        T get() throws Exception;
    }
}
