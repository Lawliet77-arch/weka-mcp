package dev.wekamcp.interop;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.Preprocessing.FilterSpec;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.experiment.ExperimentRecord;
import dev.wekamcp.experiment.ExperimentStore;
import dev.wekamcp.experiment.ModelCodec;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.ClustererEntry;
import dev.wekamcp.registry.SessionRegistry.DatasetEntry;
import dev.wekamcp.registry.SessionRegistry.ModelEntry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import tools.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.SerializedObject;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.Loader;
import weka.filters.Filter;
import weka.filters.MultiFilter;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class ModelInteropService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;
    private final ExperimentStore experiments;
    private final Set<String> serializedModelTypes = new java.util.HashSet<>();

    public ModelInteropService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks,
            ExperimentStore experiments) {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
        this.experiments = experiments;
        for (Whitelist.Kind kind : Whitelist.Kind.values()) {
            for (Whitelist.AlgorithmSpec specification : whitelist.list(kind)) {
                try {
                    for (Class<?> type = Class.forName(specification.className()); type != null;
                            type = type.getSuperclass()) {
                        serializedModelTypes.add(type.getName());
                    }
                } catch (ClassNotFoundException error) {
                    throw new IllegalArgumentException("Whitelisted model type is unavailable", error);
                }
            }
        }
    }

    public ObjectNode export(Map<String, Object> arguments) throws Exception {
        String modelId = ToolSupport.optionalString(arguments, "model_id", null);
        String clustererId = ToolSupport.optionalString(arguments, "clusterer_id", null);
        if ((modelId == null) == (clustererId == null)) {
            throw new IllegalArgumentException("Specify exactly one of model_id or clusterer_id");
        }
        String format = ToolSupport.optionalString(arguments, "format", "weka_bundle");
        if (!Set.of("weka_bundle", "raw_model", "pmml").contains(format)) {
            throw new IllegalArgumentException("format must be weka_bundle, raw_model or pmml");
        }
        boolean overwrite = ToolSupport.optionalBoolean(arguments, "overwrite", false);
        Path target = config.resolveOutputPath(ToolSupport.requiredString(arguments, "path"),
                "pmml".equals(format) ? ".pmml" : ".model");
        if (Files.exists(target) && !overwrite) {
            throw new IllegalArgumentException("Output already exists; set overwrite=true to replace it");
        }
        Object model;
        Instances header;
        String modelClass;
        String kind;
        if (modelId != null) {
            ModelEntry entry = registry.model(modelId);
            model = entry.classifier();
            header = entry.trainingHeader();
            modelClass = entry.classifierClass();
            kind = "CLASSIFIER";
        } else {
            ClustererEntry entry = registry.clusterer(clustererId);
            model = entry.clusterer();
            header = entry.trainingHeader();
            modelClass = entry.clustererClass();
            kind = "CLUSTERER";
        }
        boolean bundle = "weka_bundle".equals(format);
        boolean pmml = "pmml".equals(format);
        if (pmml) {
            if (!(model instanceof weka.core.pmml.PMMLProducer producer)) {
                throw new IllegalArgumentException(modelClass + " does not implement PMMLProducer");
            }
            writeText(target, producer.toPMML(header), overwrite);
        } else {
            writeModel(target, model, header, bundle, overwrite);
        }
        ObjectNode result = ToolSupport.object();
        result.put("path", relative(target));
        result.put("format", format);
        result.put("model_class", modelClass);
        result.put("kind", kind);
        result.put("bytes", Files.size(target));
        result.put("sha256", sha256(target));
        result.put("gui_compatible", bundle && "CLASSIFIER".equals(kind));
        result.put("header_included", bundle || pmml);
        result.put("trusted_java_serialization", !pmml);
        return result;
    }

    public ObjectNode importModel(Map<String, Object> arguments) throws Exception {
        if (!ToolSupport.optionalBoolean(arguments, "trusted", false)) {
            throw new IllegalArgumentException("trusted=true is required; only import trusted Java-serialized Weka models");
        }
        Path path = config.resolveDatasetPath(ToolSupport.requiredString(arguments, "path"));
        String format = ToolSupport.optionalString(arguments, "format", "auto");
        if (!Set.of("auto", "raw_model", "weka_bundle").contains(format)) {
            throw new IllegalArgumentException("format must be auto, raw_model or weka_bundle");
        }
        ModelCodec.StandardModelPayload payload = tasks.call(config.loadTimeoutSeconds(),
                () -> ModelCodec.readStandard(path, config.maxModelBytes(), this::allowedSerializedType));
        if ("raw_model".equals(format) && payload.headerIncluded()) {
            throw new IllegalArgumentException("raw_model must contain only one serialized model");
        }
        if ("weka_bundle".equals(format) && !payload.headerIncluded()) {
            throw new IllegalArgumentException("weka_bundle must include a training header");
        }
        DatasetEntry dataset = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        boolean persist = ToolSupport.optionalBoolean(arguments, "persist", true);
        ObjectNode metadata = ToolSupport.object();
        metadata.put("operation", "import");
        metadata.put("format", payload.headerIncluded() ? "weka_bundle" : "raw_model");
        metadata.put("path", relative(path));
        metadata.put("sha256", sha256(path));
        metadata.put("trusted", true);
        metadata.put("header_source", payload.headerIncluded() ? "bundle" : "dataset");
        if (payload.model() instanceof Classifier classifier) {
            return importClassifier(classifier, dataset, payload.trainingHeader(), arguments, metadata, persist);
        }
        return importClusterer((Clusterer) payload.model(), dataset, payload.trainingHeader(), arguments, metadata, persist);
    }

    private ObjectNode importClassifier(Classifier classifier, DatasetEntry dataset, Instances bundleHeader,
            Map<String, Object> arguments, ObjectNode metadata, boolean persist) throws Exception {
        Classifier base = classifier instanceof FilteredClassifier filtered ? filtered.getClassifier() : classifier;
        whitelist.get(Whitelist.Kind.CLASSIFIER, base.getClass().getName());
        List<FilterSpec> preprocessing = preprocessing(classifier);
        Instances header = new Instances(dataset.data());
        int requested = ToolSupport.optionalInt(arguments, "class_index", 0);
        if (requested < 0 || requested > header.numAttributes()) {
            throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
        }
        int classIndex = requested > 0 ? requested - 1
                : bundleHeader != null && bundleHeader.classIndex() >= 0 ? bundleHeader.classIndex()
                : header.classIndex() >= 0 ? header.classIndex() : header.numAttributes() - 1;
        header.setClassIndex(classIndex);
        validateHeader(bundleHeader, header);
        String[] options = ((OptionHandler) base).getOptions();
        ExperimentRecord saved = null;
        if (persist) {
            ExperimentRecord record = new ExperimentRecord();
            record.classifierClass = base.getClass().getName();
            record.options = options;
            record.preprocessing = preprocessing;
            record.classIndex = classIndex;
            record.instances = header.numInstances();
            record.attributes = header.numAttributes();
            record.protocol = "import";
            record.retrainable = false;
            record.interoperability = metadata;
            saved = experiments.create(record, header, classifier, header);
        }
        ModelEntry model = registry.registerImportedModel(base.getClass().getName(), options, classifier, header,
                preprocessing, header, saved == null ? null : saved.id);
        ObjectNode result = WekaTools.modelSummary(model);
        result.put("imported", true);
        result.put("retrainable", false);
        result.set("header_source", metadata.get("header_source"));
        return result;
    }

    private ObjectNode importClusterer(Clusterer clusterer, DatasetEntry dataset, Instances bundleHeader,
            Map<String, Object> arguments, ObjectNode metadata, boolean persist) throws Exception {
        whitelist.get(Whitelist.Kind.CLUSTERER, clusterer.getClass().getName());
        Instances header = new Instances(dataset.data());
        int requested = ToolSupport.optionalInt(arguments, "class_index", 0);
        if (requested < 0 || requested > header.numAttributes()) {
            throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
        }
        if (requested > 0) {
            header.setClassIndex(requested - 1);
        }
        int removedClassIndex = ToolSupport.optionalBoolean(arguments, "remove_class", true) ? header.classIndex() : -1;
        if (removedClassIndex >= 0) {
            Filter remove = (Filter) whitelist.create(Whitelist.Kind.FILTER,
                    "weka.filters.unsupervised.attribute.Remove",
                    new String[]{"-R", Integer.toString(removedClassIndex + 1)});
            remove.setInputFormat(header);
            header = Filter.useFilter(header, remove);
        }
        header.setClassIndex(-1);
        validateHeader(bundleHeader, header);
        Instances clusteringData = header;
        int[] assignments = new int[header.numInstances() <= config.maxPredictionRows() ? header.numInstances() : 0];
        tasks.call(config.trainTimeoutSeconds(), () -> {
            for (int index = 0; index < assignments.length; index++) {
                assignments[index] = clusterer.clusterInstance(clusteringData.instance(index));
            }
            return null;
        });
        ExperimentRecord saved = null;
        if (persist) {
            ExperimentRecord record = new ExperimentRecord();
            record.kind = "CLUSTERER";
            record.clustererClass = clusterer.getClass().getName();
            record.options = ((OptionHandler) clusterer).getOptions();
            record.classIndex = -1;
            record.instances = header.numInstances();
            record.attributes = header.numAttributes();
            record.protocol = "import";
            record.retrainable = false;
            record.interoperability = metadata;
            record.removedClassIndex = removedClassIndex;
            record.assignments = assignments;
            saved = experiments.create(record, header, clusterer, header);
        }
        ClustererEntry entry = registry.registerClusterer(clusterer.getClass().getName(),
                ((OptionHandler) clusterer).getOptions(), clusterer, header, header, assignments,
                removedClassIndex, saved == null ? null : saved.id);
        ObjectNode result = ToolSupport.object();
        result.put("clusterer_id", entry.id());
        result.put("clusterer_class", entry.clustererClass());
        result.put("num_clusters", clusterer.numberOfClusters());
        result.put("instances", header.numInstances());
        result.put("removed_class_index", removedClassIndex < 0 ? -1 : removedClassIndex + 1);
        result.put("imported", true);
        result.set("header_source", metadata.get("header_source"));
        if (saved != null) {
            result.put("experiment_id", saved.id);
        }
        return result;
    }

    public ObjectNode update(Map<String, Object> arguments) throws Exception {
        ModelEntry model = registry.model(ToolSupport.requiredString(arguments, "model_id"));
        if (!(model.classifier() instanceof UpdateableClassifier)) {
            throw new IllegalArgumentException("The model must directly implement UpdateableClassifier");
        }
        String datasetId = ToolSupport.optionalString(arguments, "dataset_id", null);
        String filename = ToolSupport.optionalString(arguments, "path", null);
        if ((datasetId == null) == (filename == null)) {
            throw new IllegalArgumentException("Specify exactly one of dataset_id or path");
        }
        int maximum = ToolSupport.optionalInt(arguments, "max_instances", config.maxInstances());
        if (maximum < 1 || maximum > config.maxInstances()) {
            throw new IllegalArgumentException("max_instances must be between 1 and maxInstances=" + config.maxInstances());
        }
        long budget = ToolSupport.optionalLong(arguments, "budgetSeconds", config.trainTimeoutSeconds());
        if (budget < 1 || budget > 86_400) {
            throw new IllegalArgumentException("budgetSeconds must be between 1 and 86400");
        }
        DatasetEntry dataset = datasetId == null ? null : registry.dataset(datasetId);
        Instances input = dataset == null ? null : WekaTools.classificationData(dataset, 0);
        if (input != null && !model.trainingHeader().equalHeaders(input)) {
            throw new IllegalArgumentException("Update dataset header is incompatible with the training header");
        }
        Path path = filename == null ? null : config.resolveDatasetPath(filename);
        Classifier updated;
        UpdateResult update;
        synchronized (model) {
            updated = (Classifier) new SerializedObject(model.classifier()).getObject();
            UpdateResult result;
            if (input != null) {
                result = tasks.call(Math.min(config.trainTimeoutSeconds(), budget),
                        () -> updateInstances(updated, input));
            } else {
                String format = ToolSupport.optionalString(arguments, "format", "auto");
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(budget);
                result = tasks.call(Math.min(config.trainTimeoutSeconds(), budget),
                        () -> updateFile(updated, path, format, model.trainingHeader(), maximum, deadline));
            }
            update = result;
        }
        if (update.updated == 0) {
            throw new IllegalArgumentException("No labeled instances were applied to the model");
        }
        ObjectNode stream = ToolSupport.object();
        stream.put("operation", "incremental_update");
        stream.put("read_instances", update.read);
        stream.put("updated_instances", update.updated);
        stream.put("skipped_missing_class", update.skippedMissingClass);
        stream.put("partial", update.partial);
        stream.put("stop_reason", update.stopReason);
        ExperimentRecord record = new ExperimentRecord();
        record.parentExperimentId = model.experimentId();
        record.classifierClass = model.classifierClass();
        record.options = model.options();
        record.preprocessing = model.preprocessing();
        record.classIndex = model.trainingHeader().classIndex();
        record.instances = update.read;
        record.attributes = model.trainingHeader().numAttributes();
        record.protocol = "incremental_update";
        record.retrainable = model.retrainable();
        record.finalModelTrainedOnFullData = false;
        record.stream = stream;
        ObjectNode source = ToolSupport.object();
        source.put("operation", "update");
        source.put("source_kind", input == null ? "file" : "dataset");
        if (input == null) {
            record.sourcePath = relative(path);
            record.datasetRole = "UPDATE_SOURCE";
            source.put("path", relative(path));
        } else {
            record.datasetRole = "UPDATE_DATASET";
            source.put("dataset_id", dataset.id());
        }
        record.interoperability = source;
        ExperimentRecord saved = experiments.create(record, input, updated, model.trainingHeader());
        Instances trainingData = model.trainingData();
        if (trainingData != null && input != null) {
            trainingData = new Instances(trainingData);
            for (Instance instance : input) {
                trainingData.add(instance);
            }
        }
        ModelEntry registered = registry.replaceModel(model, updated, trainingData, saved.id);
        ObjectNode result = WekaTools.modelSummary(registered);
        result.put("updated_instances", update.updated);
        result.put("skipped_missing_class", update.skippedMissingClass);
        result.put("read_instances", update.read);
        result.put("partial", update.partial);
        result.put("stop_reason", update.stopReason);
        result.put("experiment_id", saved.id);
        result.set("stream", stream);
        return result;
    }

    private static UpdateResult updateInstances(Classifier classifier, Instances input) throws Exception {
        int updated = 0;
        int skipped = 0;
        for (Instance instance : input) {
            if (instance.classIsMissing()) {
                skipped++;
                continue;
            }
            ((UpdateableClassifier) classifier).updateClassifier(instance);
            updated++;
        }
        return new UpdateResult(input.numInstances(), updated, skipped, false, "end_of_dataset");
    }

    private static UpdateResult updateFile(Classifier classifier, Path path, String format, Instances trainingHeader,
            int maximum, long deadline) throws Exception {
        if (!Set.of("auto", "arff", "csv").contains(format)) {
            throw new IllegalArgumentException("format must be auto, arff or csv");
        }
        String filename = path.getFileName().toString().toLowerCase(Locale.ROOT);
        boolean arff = "arff".equals(format) || "auto".equals(format) && filename.endsWith(".arff");
        if ("auto".equals(format) && !arff && !filename.endsWith(".csv")) {
            throw new IllegalArgumentException("auto format requires an .arff or .csv filename");
        }
        AbstractFileLoader fileLoader = arff ? new ArffLoader() : new CSVLoader();
        fileLoader.setFile(path.toFile());
        Loader loader = fileLoader;
        Instances header = loader.getStructure();
        if (header.numAttributes() < 1 || header.numAttributes() > 10_000) {
            throw new IllegalArgumentException("Invalid update attribute count");
        }
        header.setClassIndex(trainingHeader.classIndex());
        if (!trainingHeader.equalHeaders(header)) {
            throw new IllegalArgumentException("Update file header is incompatible with the training header");
        }
        int read = 0;
        int updated = 0;
        int skipped = 0;
        boolean partial = false;
        String stopReason = "end_of_file";
        while (read < maximum) {
            if (System.nanoTime() >= deadline) {
                partial = true;
                stopReason = "budget_exhausted";
                break;
            }
            Instance instance = loader.getNextInstance(header);
            if (instance == null) {
                break;
            }
            read++;
            if (instance.classIsMissing()) {
                skipped++;
                continue;
            }
            ((UpdateableClassifier) classifier).updateClassifier(instance);
            updated++;
        }
        if (read == maximum && "end_of_file".equals(stopReason)) {
            partial = true;
            stopReason = "max_instances";
        }
        return new UpdateResult(read, updated, skipped, partial, stopReason);
    }

    private List<FilterSpec> preprocessing(Classifier classifier) {
        if (!(classifier instanceof FilteredClassifier filtered)) {
            return List.of();
        }
        Filter filter = filtered.getFilter();
        Filter[] filters = filter instanceof MultiFilter multi ? multi.getFilters() : new Filter[]{filter};
        List<FilterSpec> result = new ArrayList<>();
        for (Filter item : filters) {
            whitelist.get(Whitelist.Kind.FILTER, item.getClass().getName());
            result.add(new FilterSpec(item.getClass().getName(), item.getOptions()));
        }
        return List.copyOf(result);
    }

    private boolean allowedSerializedType(String name) {
        if (name.startsWith("weka.core.") || name.startsWith("weka.estimators.")
                || name.startsWith("weka.classifiers.trees.j48.")
                || name.startsWith("weka.classifiers.trees.m5.")
                || name.startsWith("weka.classifiers.bayes.net.")
                || name.startsWith("weka.classifiers.functions.supportVector.")
                || name.startsWith("weka.classifiers.functions.neural.")
                || name.equals("weka.classifiers.meta.FilteredClassifier")
                || name.equals("weka.classifiers.rules.DecisionTableHashKey")
                || name.equals("weka.filters.unsupervised.attribute.PotentialClassIgnorer")
                || name.equals("weka.filters.Filter") || name.equals("weka.filters.MultiFilter")
                || serializedModelTypes.contains(name)) {
            return true;
        }
        for (Whitelist.Kind kind : Whitelist.Kind.values()) {
            for (Whitelist.AlgorithmSpec specification : whitelist.list(kind)) {
                if (name.equals(specification.className()) || name.startsWith(specification.className() + "$")) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void validateHeader(Instances bundleHeader, Instances datasetHeader) {
        if (bundleHeader != null) {
            bundleHeader.setClassIndex(datasetHeader.classIndex());
            if (!bundleHeader.equalHeaders(datasetHeader)) {
                throw new IllegalArgumentException("Model training header is incompatible with the supplied dataset");
            }
        }
    }

    private static void writeText(Path target, String content, boolean overwrite) throws IOException {
        Path temporary = Files.createTempFile(target.getParent(), ".weka-export-", ".tmp");
        try {
            Files.writeString(temporary, content, java.nio.charset.StandardCharsets.UTF_8);
            if (overwrite) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } else {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(temporary);
            throw e;
        }
    }

    private static void writeModel(Path target, Object model, Instances header, boolean bundle, boolean overwrite)
            throws IOException {
        Path temporary = Files.createTempFile(target.getParent(), ".weka-export-", ".tmp");
        try {
            try (ObjectOutputStream output = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(temporary)))) {
                output.writeObject(model);
                if (bundle) {
                    output.writeObject(header.stringFreeStructure());
                }
            }
            if (overwrite) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } else {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(temporary);
            throw e;
        }
    }

    private String relative(Path path) {
        return config.dataRoot().relativize(path.toAbsolutePath().normalize()).toString().replace('\\', '/');
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = Files.newInputStream(path)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    digest.update(buffer, 0, read);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private record UpdateResult(int read, int updated, int skippedMissingClass, boolean partial, String stopReason) {
    }
}
