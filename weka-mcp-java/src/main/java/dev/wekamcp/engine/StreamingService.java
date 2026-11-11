package dev.wekamcp.engine;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.experiment.ExperimentRecord;
import dev.wekamcp.experiment.ExperimentStore;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.ModelEntry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import tools.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.Loader;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class StreamingService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;
    private final ExperimentStore experiments;

    public StreamingService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks,
            ExperimentStore experiments) {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
        this.experiments = experiments;
    }

    public ObjectNode train(Map<String, Object> arguments) throws Exception {
        Path path = config.resolveDatasetPath(ToolSupport.requiredString(arguments, "path"));
        String format = ToolSupport.optionalString(arguments, "format", "auto");
        if (!Set.of("auto", "arff", "csv").contains(format)) {
            throw new IllegalArgumentException("format must be auto, arff or csv");
        }
        String filename = path.getFileName().toString().toLowerCase(java.util.Locale.ROOT);
        if ("auto".equals(format) && !filename.endsWith(".arff") && !filename.endsWith(".csv")) {
            throw new IllegalArgumentException("auto format requires an .arff or .csv filename");
        }
        String className = ToolSupport.requiredString(arguments, "classifier_class");
        String[] options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
        Classifier created = (Classifier) whitelist.create(Whitelist.Kind.CLASSIFIER, className, options);
        if (!(created instanceof UpdateableClassifier)) {
            throw new IllegalArgumentException("classifier_class must implement UpdateableClassifier");
        }
        String evaluationMode = ToolSupport.optionalString(arguments, "evaluation", "holdout");
        if (!Set.of("holdout", "prequential").contains(evaluationMode)) {
            throw new IllegalArgumentException("evaluation must be holdout or prequential");
        }
        int maxInstances = ToolSupport.optionalInt(arguments, "max_instances", config.maxInstances());
        int maxTestInstances = ToolSupport.optionalInt(arguments, "max_test_instances",
                Math.min(1000, config.maxPredictionRows()));
        int testEvery = ToolSupport.optionalInt(arguments, "test_every", 5);
        long budgetSeconds = ToolSupport.optionalLong(arguments, "budgetSeconds", config.trainTimeoutSeconds());
        if (maxInstances < 1 || maxInstances > config.maxInstances()) {
            throw new IllegalArgumentException("max_instances must be between 1 and maxInstances=" + config.maxInstances());
        }
        if (maxTestInstances < 0 || maxTestInstances > config.maxPredictionRows()) {
            throw new IllegalArgumentException(
                    "max_test_instances must be between 0 and maxPredictionRows=" + config.maxPredictionRows());
        }
        if (testEvery < 1 || testEvery > 1_000_000) {
            throw new IllegalArgumentException("test_every must be between 1 and 1000000");
        }
        if (budgetSeconds < 1 || budgetSeconds > 86_400) {
            throw new IllegalArgumentException("budgetSeconds must be between 1 and 86400");
        }
        boolean persist = ToolSupport.optionalBoolean(arguments, "persist", true);
        long started = System.nanoTime();
        long deadline = started + TimeUnit.SECONDS.toNanos(budgetSeconds);
        StreamResult streamed = tasks.call(Math.min(config.trainTimeoutSeconds(), budgetSeconds), () -> trainIncrementally(
                path, format, created, arguments, evaluationMode, maxInstances, maxTestInstances, testEvery, deadline));
        if (streamed.trainingInstances == 0) {
            throw new IllegalArgumentException("No labeled training instances were read");
        }

        ObjectNode streamMetadata = ToolSupport.object();
        streamMetadata.put("evaluation", evaluationMode);
        streamMetadata.put("test_every", testEvery);
        streamMetadata.put("max_instances", maxInstances);
        streamMetadata.put("max_test_instances", maxTestInstances);
        streamMetadata.put("budgetSeconds", budgetSeconds);
        streamMetadata.put("read_instances", streamed.readInstances);
        streamMetadata.put("training_instances", streamed.trainingInstances);
        streamMetadata.put("test_instances", streamed.testInstances);
        streamMetadata.put("skipped_missing_class", streamed.skippedMissingClass);
        streamMetadata.put("stop_reason", streamed.stopReason);
        streamMetadata.put("partial", streamed.partial);
        ExperimentRecord saved = null;
        if (persist) {
            ExperimentRecord record = new ExperimentRecord();
            record.kind = "CLASSIFIER";
            record.classifierClass = className;
            record.options = options;
            record.classIndex = streamed.header.classIndex();
            record.instances = streamed.readInstances;
            record.attributes = streamed.header.numAttributes();
            record.protocol = "stream";
            record.sourcePath = config.dataRoot().relativize(path).toString().replace('\\', '/');
            record.stream = streamMetadata;
            record.testMetrics = streamed.metrics;
            record.finalModelTrainedOnFullData = streamed.testInstances == 0 || "prequential".equals(evaluationMode);
            if (streamed.metrics != null) {
                record.selectionMetric = streamed.header.classAttribute().isNominal() ? "accuracy" : "rmse";
                record.optimize = record.selectionMetric;
                String scoreName = streamed.header.classAttribute().isNominal() ? "accuracy_percent" : "rmse";
                if (streamed.metrics.has(scoreName)) {
                    record.testScore = streamed.metrics.get(scoreName).doubleValue();
                }
            }
            saved = experiments.create(record, null, created, streamed.header);
        }
        ModelEntry model = registry.registerModel(className, options, created, streamed.header,
                java.util.List.of(), null, saved == null ? null : saved.id);
        ObjectNode result = WekaTools.modelSummary(model);
        result.put("source_path", config.dataRoot().relativize(path).toString().replace('\\', '/'));
        result.put("model_status", "TRAINED");
        result.set("stream", streamMetadata);
        result.put("read_instances", streamed.readInstances);
        result.put("training_instances", streamed.trainingInstances);
        result.put("test_instances", streamed.testInstances);
        result.put("skipped_missing_class", streamed.skippedMissingClass);
        result.put("stop_reason", streamed.stopReason);
        result.put("partial", streamed.partial);
        result.set("evaluation", streamed.metrics);
        result.set("test", streamed.metrics);
        result.put("elapsed_ms", (System.nanoTime() - started) / 1_000_000);
        return result;
    }

    private StreamResult trainIncrementally(Path path, String format, Classifier classifier, Map<String, Object> arguments,
            String evaluationMode, int maxInstances, int maxTestInstances, int testEvery, long deadline) throws Exception {
        try (DatasetIo.LoaderResource opened = DatasetIo.open(path, format)) {
            Loader loader = opened.loader();
            Instances header = loader.getStructure();
            if (header.numAttributes() < 2 || header.numAttributes() > config.maxAttributes()) {
                throw new IllegalArgumentException(
                        "A stream requires 2 to maxAttributes=" + config.maxAttributes() + " attributes");
            }
            int classIndex = ToolSupport.optionalInt(arguments, "class_index", 0);
            if (classIndex < 0 || classIndex > header.numAttributes()) {
                throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
            }
            header.setClassIndex(classIndex == 0 ? header.numAttributes() - 1 : classIndex - 1);
            if (!header.classAttribute().isNominal() && !header.classAttribute().isNumeric()) {
                throw new IllegalArgumentException("The stream class attribute must be nominal or numeric");
            }
            classifier.buildClassifier(new Instances(header, 0));
            UpdateableClassifier updateable = (UpdateableClassifier) classifier;
            Evaluation evaluation = new Evaluation(header);
            evaluation.setDiscardPredictions(true);
            Instances test = new Instances(header, 0);
            int read = 0;
            int trained = 0;
            int tested = 0;
            int skippedMissingClass = 0;
            String stopReason = "end_of_file";
            boolean partial = false;
            while (read < maxInstances) {
                if (System.nanoTime() >= deadline) {
                    stopReason = "budget_exhausted";
                    partial = true;
                    break;
                }
                Instance instance = loader.getNextInstance(header);
                if (instance == null) {
                    break;
                }
                read++;
                if (instance.classIsMissing()) {
                    skippedMissingClass++;
                    continue;
                }
                if ("holdout".equals(evaluationMode) && read % testEvery == 0 && test.numInstances() < maxTestInstances) {
                    test.add(instance);
                    tested++;
                    continue;
                }
                if ("prequential".equals(evaluationMode) && read % testEvery == 0 && tested < maxTestInstances) {
                    evaluation.evaluateModelOnce(classifier, instance);
                    tested++;
                }
                updateable.updateClassifier(instance);
                trained++;
            }
            if (read == maxInstances && "end_of_file".equals(stopReason)) {
                stopReason = "max_instances";
                partial = true;
            }
            if ("holdout".equals(evaluationMode)) {
                for (Instance instance : test) {
                    evaluation.evaluateModelOnce(classifier, instance);
                }
            }
            ObjectNode metrics = null;
            if (tested > 0) {
                metrics = ToolSupport.object();
                metrics.put("mode", evaluationMode);
                metrics.put("instances", tested);
                metrics.put("test_then_train", "prequential".equals(evaluationMode));
                WekaTools.addEvaluationMetrics(metrics, evaluation, header);
            }
            return new StreamResult(header, read, trained, tested, skippedMissingClass, stopReason, partial, metrics);
        }
    }

    private record StreamResult(Instances header, int readInstances, int trainingInstances, int testInstances,
            int skippedMissingClass, String stopReason, boolean partial, ObjectNode metrics) {
    }
}
