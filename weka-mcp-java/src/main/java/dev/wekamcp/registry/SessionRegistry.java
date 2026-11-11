package dev.wekamcp.registry;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.engine.Preprocessing.FilterSpec;
import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instances;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class SessionRegistry {
    public record DatasetEntry(String id, String name, Instances data) {
    }

    public record ModelEntry(
            String id,
            String classifierClass,
            String[] options,
            Classifier classifier,
            Instances trainingHeader,
            List<FilterSpec> preprocessing,
            Instances trainingData,
            String experimentId,
            boolean retrainable) {
        public ModelEntry(String id, String classifierClass, String[] options, Classifier classifier,
                Instances trainingHeader, List<FilterSpec> preprocessing, Instances trainingData, String experimentId) {
            this(id, classifierClass, options, classifier, trainingHeader, preprocessing, trainingData, experimentId,
                    true);
        }

        public ModelEntry asImported() {
            return new ModelEntry(id, classifierClass, options, classifier, trainingHeader, preprocessing,
                    trainingData, experimentId, false);
        }
        public ModelEntry(
                String id, String classifierClass, String[] options, Classifier classifier, Instances trainingHeader) {
            this(id, classifierClass, options, classifier, trainingHeader, List.of(), trainingHeader, null);
        }

        public ModelEntry(
                String id, String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
                List<FilterSpec> preprocessing) {
            this(id, classifierClass, options, classifier, trainingHeader, preprocessing, trainingHeader, null);
        }

        public ModelEntry(
                String id, String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
                List<FilterSpec> preprocessing, String experimentId) {
            this(id, classifierClass, options, classifier, trainingHeader, preprocessing, trainingHeader, experimentId);
        }

        public ModelEntry {
            options = options.clone();
            trainingData = trainingData == null || trainingData.numInstances() == 0 ? null : trainingData;
            trainingHeader = new Instances(trainingHeader, 0);
            preprocessing = List.copyOf(preprocessing);
        }

        @Override
        public String[] options() {
            return options.clone();
        }
    }

    public record ClustererEntry(
            String id,
            String clustererClass,
            String[] options,
            Clusterer clusterer,
            Instances trainingHeader,
            Instances trainingData,
            int[] assignments,
            int removedClassIndex,
            String experimentId) {
        public ClustererEntry(
                String id, String clustererClass, String[] options, Clusterer clusterer,
                Instances trainingHeader, int[] assignments, int removedClassIndex) {
            this(id, clustererClass, options, clusterer, trainingHeader, trainingHeader, assignments,
                    removedClassIndex, null);
        }

        public ClustererEntry(
                String id, String clustererClass, String[] options, Clusterer clusterer,
                Instances trainingHeader, int[] assignments) {
            this(id, clustererClass, options, clusterer, trainingHeader, assignments, -1);
        }

        public ClustererEntry {
            options = options.clone();
            trainingData = trainingData == null || trainingData.numInstances() == 0 ? null : trainingData;
            trainingHeader = new Instances(trainingHeader, 0);
            assignments = assignments.clone();
        }

        @Override
        public String[] options() {
            return options.clone();
        }

        @Override
        public int[] assignments() {
            return assignments.clone();
        }
    }

    private final ServerConfig config;
    private final Map<String, DatasetEntry> datasets = new LinkedHashMap<>();
    private final Map<String, ModelEntry> models = new LinkedHashMap<>();
    private final Map<String, ClustererEntry> clusterers = new LinkedHashMap<>();

    public SessionRegistry(ServerConfig config) {
        this.config = config;
    }

    public synchronized DatasetEntry registerDataset(String name, Instances data) {
        requireCapacity(datasets.size(), config.maxDatasets(), "datasets");
        String id = newId();
        DatasetEntry entry = new DatasetEntry(id, name, data);
        datasets.put(id, entry);
        return entry;
    }

    public synchronized ModelEntry registerModel(
            String classifierClass, String[] options, Classifier classifier, Instances trainingHeader) {
        return registerModel(classifierClass, options, classifier, trainingHeader, List.of());
    }

    public synchronized ModelEntry registerModel(
            String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
            List<FilterSpec> preprocessing) {
        return registerModel(classifierClass, options, classifier, trainingHeader, preprocessing, null);
    }

    public synchronized ModelEntry registerModel(
            String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
            List<FilterSpec> preprocessing, String experimentId) {
        return registerModel(classifierClass, options, classifier, trainingHeader, preprocessing, trainingHeader,
                experimentId);
    }

    public synchronized ModelEntry registerModel(
            String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
            List<FilterSpec> preprocessing, Instances trainingData, String experimentId) {
        requireCapacity(models.size(), config.maxModels(), "models");
        String id = newId();
        ModelEntry entry = new ModelEntry(
                id, classifierClass, options, classifier, trainingHeader, preprocessing, trainingData, experimentId);
        models.put(id, entry);
        return entry;
    }

    public synchronized ModelEntry registerImportedModel(
            String classifierClass, String[] options, Classifier classifier, Instances trainingHeader,
            List<FilterSpec> preprocessing, Instances trainingData, String experimentId) {
        ModelEntry entry = registerModel(classifierClass, options, classifier, trainingHeader,
                preprocessing, trainingData, experimentId);
        ModelEntry imported = entry.asImported();
        models.put(entry.id(), imported);
        return imported;
    }

    public synchronized ModelEntry replaceModel(ModelEntry previous, Classifier classifier,
            Instances trainingData, String experimentId) {
        ModelEntry entry = new ModelEntry(previous.id(), previous.classifierClass(), previous.options(), classifier,
                previous.trainingHeader(), previous.preprocessing(), trainingData, experimentId, previous.retrainable());
        models.put(previous.id(), entry);
        return entry;
    }

    public synchronized ClustererEntry registerClusterer(
            String clustererClass, String[] options, Clusterer clusterer, Instances trainingHeader, int[] assignments) {
        return registerClusterer(clustererClass, options, clusterer, trainingHeader, assignments, -1);
    }

    public synchronized ClustererEntry registerClusterer(
            String clustererClass, String[] options, Clusterer clusterer, Instances trainingHeader, int[] assignments,
            int removedClassIndex) {
        return registerClusterer(clustererClass, options, clusterer, trainingHeader, trainingHeader, assignments,
                removedClassIndex, null);
    }

    public synchronized ClustererEntry registerClusterer(
            String clustererClass, String[] options, Clusterer clusterer, Instances trainingHeader,
            Instances trainingData, int[] assignments, int removedClassIndex, String experimentId) {
        requireCapacity(clusterers.size(), config.maxClusterers(), "clusterers");
        String id = newId();
        ClustererEntry entry = new ClustererEntry(
                id, clustererClass, options, clusterer, trainingHeader, trainingData, assignments,
                removedClassIndex, experimentId);
        clusterers.put(id, entry);
        return entry;
    }

    public synchronized DatasetEntry dataset(String id) {
        DatasetEntry entry = datasets.get(validId(id, "dataset_id"));
        if (entry == null) {
            throw new IllegalArgumentException("Unknown dataset_id: " + id);
        }
        return entry;
    }

    public synchronized ModelEntry model(String id) {
        ModelEntry entry = models.get(validId(id, "model_id"));
        if (entry == null) {
            throw new IllegalArgumentException("Unknown model_id: " + id);
        }
        return entry;
    }

    public synchronized ClustererEntry clusterer(String id) {
        ClustererEntry entry = clusterers.get(validId(id, "clusterer_id"));
        if (entry == null) {
            throw new IllegalArgumentException("Unknown clusterer_id: " + id);
        }
        return entry;
    }

    public synchronized List<DatasetEntry> datasets() {
        return List.copyOf(datasets.values());
    }

    public synchronized List<ModelEntry> models() {
        return List.copyOf(models.values());
    }

    public synchronized List<ClustererEntry> clusterers() {
        return List.copyOf(clusterers.values());
    }

    public synchronized String release(String id) {
        String validId = validId(id, "resource_id");
        if (datasets.remove(validId) != null) {
            return "dataset";
        }
        if (models.remove(validId) != null) {
            return "model";
        }
        if (clusterers.remove(validId) != null) {
            return "clusterer";
        }
        throw new IllegalArgumentException("Unknown resource_id: " + id);
    }

    private static String validId(String id, String name) {
        if (id == null) {
            throw new IllegalArgumentException(name + " is required");
        }
        try {
            UUID.fromString(id);
            return id;
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(name + " must be a UUID", e);
        }
    }

    private static String newId() {
        return UUID.randomUUID().toString();
    }

    private static void requireCapacity(int size, int maximum, String resource) {
        if (size >= maximum) {
            throw new IllegalStateException("Maximum number of " + resource + " reached: " + maximum
                    + "; release an unused resource with weka_release");
        }
    }
}
