package dev.wekamcp.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

public record ServerConfig(
        Path dataRoot,
        Path whitelistPath,
        int maxDatasets,
        int maxModels,
        int maxClusterers,
        int maxInstances,
        int maxAttributes,
        long maxFileBytes,
        int maxOptionsLength,
        int maxOptionTokenLength,
        int maxInstanceJsonBytes,
        int maxPredictionRows,
        int maxOutputBytes,
        long trainTimeoutSeconds,
        long loadTimeoutSeconds,
        int maxConcurrentTasks,
        int maxQueuedTasks,
        Path experimentRoot,
        Path searchSpacePath,
        int maxExperiments,
        long maxExperimentDiskBytes,
        long maxModelBytes,
        long autoMlBudgetSeconds) {

    private static final String[] KEYS = {
            "dataRoot", "whitelistPath", "maxDatasets", "maxModels", "maxClusterers",
            "maxInstances", "maxAttributes", "maxFileBytes", "maxOptionsLength",
            "maxOptionTokenLength", "maxInstanceJsonBytes", "maxPredictionRows", "maxOutputBytes",
            "trainTimeoutSeconds", "loadTimeoutSeconds", "maxConcurrentTasks", "maxQueuedTasks",
            "experimentRoot", "searchSpacePath", "maxExperiments", "maxExperimentDiskBytes",
            "maxModelBytes", "autoMlBudgetSeconds"
    };

    public ServerConfig {
        requirePositive(maxDatasets, "maxDatasets");
        requirePositive(maxModels, "maxModels");
        requirePositive(maxClusterers, "maxClusterers");
        requirePositive(maxInstances, "maxInstances");
        requirePositive(maxAttributes, "maxAttributes");
        requirePositive(maxFileBytes, "maxFileBytes");
        requirePositive(maxOptionsLength, "maxOptionsLength");
        requirePositive(maxOptionTokenLength, "maxOptionTokenLength");
        requirePositive(maxInstanceJsonBytes, "maxInstanceJsonBytes");
        requirePositive(maxPredictionRows, "maxPredictionRows");
        requirePositive(maxOutputBytes, "maxOutputBytes");
        requirePositive(trainTimeoutSeconds, "trainTimeoutSeconds");
        requirePositive(loadTimeoutSeconds, "loadTimeoutSeconds");
        requirePositive(maxConcurrentTasks, "maxConcurrentTasks");
        requirePositive(maxExperiments, "maxExperiments");
        requirePositive(maxExperimentDiskBytes, "maxExperimentDiskBytes");
        requirePositive(maxModelBytes, "maxModelBytes");
        requirePositive(autoMlBudgetSeconds, "autoMlBudgetSeconds");
        if (maxQueuedTasks < 0) {
            throw new IllegalArgumentException("maxQueuedTasks must be zero or greater");
        }
    }

    public static ServerConfig load() throws IOException {
        String configured = System.getProperty("weka.mcp.config");
        if (configured == null || configured.isBlank()) {
            configured = System.getenv("WEKA_MCP_CONFIG");
        }
        Path path = configured == null || configured.isBlank()
                ? Path.of("config.properties").toAbsolutePath().normalize()
                : Path.of(configured).toAbsolutePath().normalize();
        return load(path);
    }

    public static ServerConfig load(Path configPath) throws IOException {
        Properties properties = new Properties();
        if (Files.exists(configPath)) {
            try (var reader = Files.newBufferedReader(configPath, java.nio.charset.StandardCharsets.UTF_8)) {
                properties.load(reader);
            }
        }
        for (String key : KEYS) {
            String environmentValue = System.getenv(environmentKey(key));
            if (environmentValue != null && !environmentValue.isBlank()) {
                properties.setProperty(key, environmentValue);
            }
        }

        Path base = configPath.toAbsolutePath().normalize().getParent();
        Path dataRoot = resolve(base, properties.getProperty("dataRoot", "data"));
        Files.createDirectories(dataRoot);
        dataRoot = dataRoot.toRealPath();
        Path whitelistPath = resolve(base, properties.getProperty("whitelistPath", "weka-whitelist.yaml"));
        Path experimentRoot = properties.containsKey("experimentRoot")
                ? resolve(base, properties.getProperty("experimentRoot"))
                : dataRoot.resolve("experiments");
        Path searchSpacePath = resolve(base, properties.getProperty("searchSpacePath", "automl-search.yaml"));

        return new ServerConfig(
                dataRoot,
                whitelistPath,
                intValue(properties, "maxDatasets", 20),
                intValue(properties, "maxModels", 50),
                intValue(properties, "maxClusterers", 20),
                intValue(properties, "maxInstances", 100_000),
                intValue(properties, "maxAttributes", 10_000),
                longValue(properties, "maxFileBytes", 104_857_600L),
                intValue(properties, "maxOptionsLength", 20),
                intValue(properties, "maxOptionTokenLength", 64),
                intValue(properties, "maxInstanceJsonBytes", 16_384),
                intValue(properties, "maxPredictionRows", 1_000),
                intValue(properties, "maxOutputBytes", 1_048_576),
                longValue(properties, "trainTimeoutSeconds", 120L),
                longValue(properties, "loadTimeoutSeconds", 30L),
                intValue(properties, "maxConcurrentTasks", 2),
                intValue(properties, "maxQueuedTasks", 8),
                experimentRoot,
                searchSpacePath,
                intValue(properties, "maxExperiments", 100),
                longValue(properties, "maxExperimentDiskBytes", 536_870_912L),
                longValue(properties, "maxModelBytes", 67_108_864L),
                longValue(properties, "autoMlBudgetSeconds", 300L));
    }

    public Path resolveDatasetPath(String userPath) throws IOException {
        if (userPath == null || userPath.isBlank()) {
            throw new IllegalArgumentException("path is required");
        }
        Path requested = Path.of(userPath);
        if (!requested.isAbsolute()) {
            requested = dataRoot.resolve(requested);
        }
        requested = requested.toRealPath();
        if (!requested.startsWith(dataRoot) || !Files.isRegularFile(requested)) {
            throw new IllegalArgumentException("path must identify a regular file inside the configured data root");
        }
        long size = Files.size(requested);
        if (size > maxFileBytes) {
            throw new IllegalArgumentException("file exceeds maxFileBytes=" + maxFileBytes);
        }
        return requested;
    }

    public Path resolveOutputPath(String userPath, String extension) throws IOException {
        if (userPath == null || userPath.isBlank()) {
            throw new IllegalArgumentException("path is required");
        }
        Path requested = Path.of(userPath);
        if (!requested.isAbsolute()) {
            requested = dataRoot.resolve(requested);
        }
        requested = requested.toAbsolutePath().normalize();
        if (!requested.startsWith(dataRoot) || !requested.getFileName().toString().endsWith(extension)) {
            throw new IllegalArgumentException("path must stay inside the configured data root and end with " + extension);
        }
        Path parent = requested.getParent();
        Files.createDirectories(parent);
        Path realParent = parent.toRealPath();
        if (!realParent.startsWith(dataRoot)) {
            throw new IllegalArgumentException("output path escapes the configured data root");
        }
        Path target = realParent.resolve(requested.getFileName().toString()).normalize();
        if (Files.isSymbolicLink(target) || Files.exists(target) && !Files.isRegularFile(target)) {
            throw new IllegalArgumentException("output path must identify a regular file and must not be a symlink");
        }
        return target;
    }

    private static Path resolve(Path base, String value) {
        Path path = Path.of(value);
        return path.isAbsolute() ? path.normalize() : base.resolve(path).normalize();
    }

    private static int intValue(Properties properties, String key, int defaultValue) {
        return Integer.parseInt(properties.getProperty(key, Integer.toString(defaultValue)).trim());
    }

    private static long longValue(Properties properties, String key, long defaultValue) {
        return Long.parseLong(properties.getProperty(key, Long.toString(defaultValue)).trim());
    }

    private static String environmentKey(String key) {
        return "WEKA_MCP_" + key.replaceAll("([a-z])([A-Z])", "$1_$2").toUpperCase();
    }

    private static void requirePositive(long value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
    }
}
