package dev.wekamcp.experiment;

import dev.wekamcp.config.ServerConfig;
import tools.jackson.databind.json.JsonMapper;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;

public final class ExperimentStore {
    private final ServerConfig config;
    private final JsonMapper json = new JsonMapper();
    private final Path root;

    public ExperimentStore(ServerConfig config) throws IOException {
        this.config = config;
        Path configuredRoot = config.experimentRoot().toAbsolutePath().normalize();
        Files.createDirectories(configuredRoot);
        this.root = configuredRoot.toRealPath();
    }

    public synchronized ExperimentRecord create(
            ExperimentRecord record, Instances dataset, Object model, Instances trainingHeader) throws IOException {
        if (listIds().size() >= config.maxExperiments()) {
            throw new IllegalStateException("Maximum number of experiments reached: " + config.maxExperiments());
        }
        String kind = record.kind == null ? "CLASSIFIER" : record.kind;
        if (!kind.equals("CLASSIFIER") && !kind.equals("CLUSTERER")) {
            throw new IllegalArgumentException("Experiment kind must be CLASSIFIER or CLUSTERER");
        }
        record.kind = kind;
        if (kind.equals("CLASSIFIER") != (model instanceof Classifier)
                || kind.equals("CLUSTERER") != (model instanceof weka.clusterers.Clusterer)) {
            throw new IllegalArgumentException("Experiment kind does not match the model");
        }
        if (dataset == null && (record.sourcePath == null || record.sourcePath.isBlank())) {
            throw new IllegalArgumentException("A dataset or sourcePath is required");
        }
        record.id = record.id == null ? UUID.randomUUID().toString() : validId(record.id);
        Path destination = root.resolve(record.id);
        if (Files.exists(destination)) {
            throw new IllegalArgumentException("Experiment already exists: " + record.id);
        }
        Path temporary = Files.createTempDirectory(root, ".tmp-");
        try {
            Path datasetFile = temporary.resolve("dataset.arff");
            if (dataset != null) {
                ArffSaver saver = new ArffSaver();
                saver.setInstances(dataset);
                saver.setFile(datasetFile.toFile());
                saver.writeBatch();
                record.dataHash = sha256(datasetFile);
            } else {
                record.dataHash = sha256(config.resolveDatasetPath(record.sourcePath));
            }

            Path modelFile = temporary.resolve("model.model");
            ModelCodec.write(modelFile, model, trainingHeader);
            if (Files.size(modelFile) > config.maxModelBytes()) {
                throw new IllegalArgumentException("Model exceeds maxModelBytes=" + config.maxModelBytes());
            }
            record.headerHash = sha256(trainingHeader.toString());
            record.createdAt = java.time.Instant.now().toString();
            record.javaVersion = System.getProperty("java.version");
            record.wekaVersion = weka.core.Version.VERSION;
            record.status = "COMPLETE";
            json.writerWithDefaultPrettyPrinter().writeValue(temporary.resolve("experiment.json").toFile(), record);
            if (directorySize(root) + directorySize(temporary) > config.maxExperimentDiskBytes()) {
                throw new IllegalStateException("Experiment disk quota exceeded: " + config.maxExperimentDiskBytes());
            }
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
            return record;
        } catch (IOException | RuntimeException e) {
            deleteRecursively(temporary);
            throw e;
        }
    }

    public ExperimentRecord get(String id) throws IOException {
        Path directory = experimentDirectory(id);
        Path manifest = directory.resolve("experiment.json");
        if (!Files.isRegularFile(manifest)) {
            throw new IllegalArgumentException("Unknown or incomplete experiment: " + id);
        }
        ExperimentRecord record = json.readValue(manifest.toFile(), ExperimentRecord.class);
        if (record.schemaVersion != 1 || !id.equals(record.id)) {
            throw new IllegalArgumentException("Unsupported or inconsistent experiment manifest");
        }
        return record;
    }

    public List<ExperimentRecord> list(int offset, int limit) throws IOException {
        if (offset < 0 || limit < 1 || limit > 100) {
            throw new IllegalArgumentException("offset must be >= 0 and limit must be between 1 and 100");
        }
        List<ExperimentRecord> records = new ArrayList<>();
        for (String id : listIds()) {
            records.add(get(id));
        }
        records.sort(Comparator.comparing((ExperimentRecord record) -> record.createdAt).reversed());
        if (offset >= records.size()) {
            return List.of();
        }
        return List.copyOf(records.subList(offset, Math.min(records.size(), offset + limit)));
    }

    public int count() throws IOException {
        return listIds().size();
    }

    public ModelCodec.ModelPayload loadModel(String id) throws IOException, ClassNotFoundException {
        ExperimentRecord record = get(id);
        Path modelFile = experimentDirectory(id).resolve("model.model");
        if (!Files.isRegularFile(modelFile)) {
            throw new IllegalArgumentException("Experiment model is missing: " + id);
        }
        boolean clusterer = "CLUSTERER".equals(record.kind);
        String expectedClass = clusterer ? record.clustererClass : record.classifierClass;
        return ModelCodec.read(modelFile, config.maxModelBytes(), expectedClass,
                !clusterer && !record.preprocessing.isEmpty(), clusterer);
    }

    public Instances loadDataset(String id) throws Exception {
        ExperimentRecord record = get(id);
        Path datasetFile = experimentDirectory(id).resolve("dataset.arff");
        if (Files.isRegularFile(datasetFile)) {
            ArffLoader loader = new ArffLoader();
            loader.setFile(datasetFile.toFile());
            return loader.getDataSet();
        }
        if (record.sourcePath == null) {
            throw new IllegalArgumentException("Experiment dataset is missing: " + id);
        }
        Instances data = new weka.core.converters.ConverterUtils.DataSource(
                config.resolveDatasetPath(record.sourcePath).toString()).getDataSet();
        data.setClassIndex(record.classIndex);
        return data;
    }

    public void delete(String id) throws IOException {
        Path directory = experimentDirectory(id);
        if (!Files.isDirectory(directory)) {
            throw new IllegalArgumentException("Unknown experiment: " + id);
        }
        deleteRecursively(directory);
    }

    private Path experimentDirectory(String id) throws IOException {
        String valid = validId(id);
        Path directory = root.resolve(valid).normalize();
        if (!directory.startsWith(root)) {
            throw new IllegalArgumentException("Invalid experiment path");
        }
        if (Files.exists(directory)) {
            Path realDirectory = directory.toRealPath();
            if (!realDirectory.startsWith(root)) {
                throw new IllegalArgumentException("Experiment path escapes the experiment root");
            }
            return realDirectory;
        }
        return directory;
    }

    private List<String> listIds() throws IOException {
        try (var paths = Files.list(root)) {
            return paths.filter(Files::isDirectory)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> {
                        try {
                            return UUID.fromString(name) != null;
                        } catch (IllegalArgumentException e) {
                            return false;
                        }
                    })
                    .sorted()
                    .toList();
        }
    }

    private static String validId(String id) {
        if (id == null) {
            throw new IllegalArgumentException("experiment_id is required");
        }
        try {
            return UUID.fromString(id).toString();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("experiment_id must be a UUID", e);
        }
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

    private static String sha256(String value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static long directorySize(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return 0;
        }
        try (var paths = Files.walk(directory)) {
            return paths.filter(Files::isRegularFile).mapToLong(path -> {
                try {
                    return Files.size(path);
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            }).sum();
        }
    }

    private static void deleteRecursively(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }
        try (var paths = Files.walk(directory)) {
            paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            });
        }
    }
}
