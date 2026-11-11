package weka.mcp.gui;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Properties;
import java.util.UUID;

public final class PluginWorkspace {
    private final Path packageDirectory;
    private final Path workspaceDirectory;
    private final Path dataDirectory;
    private final Path serviceJar;
    private final Path configFile;

    public PluginWorkspace(Path packageDirectory, Path workspaceDirectory) throws IOException {
        this.packageDirectory = packageDirectory.toAbsolutePath().normalize();
        Path workspace = workspaceDirectory.toAbsolutePath().normalize();
        Files.createDirectories(workspace);
        this.workspaceDirectory = workspace.toRealPath();
        this.dataDirectory = this.workspaceDirectory.resolve("data");
        Files.createDirectories(dataDirectory);
        this.serviceJar = this.packageDirectory.resolve("tools/weka-mcp-java-1.6.0.jar");
        this.configFile = this.workspaceDirectory.resolve("config.properties");
    }

    public static PluginWorkspace installed() throws IOException {
        String home = System.getenv("WEKA_HOME");
        Path wekaHome = home == null || home.isBlank()
                ? Path.of(System.getProperty("user.home"), "wekafiles") : Path.of(home);
        Path packageDirectory = Path.of(System.getProperty("weka.mcp.gui.package",
                wekaHome.resolve("packages/wekaMcp").toString()));
        Path workspace = Path.of(System.getProperty("weka.mcp.gui.workspace",
                Path.of(System.getProperty("user.home"), ".weka-mcp-gui").toString()));
        return new PluginWorkspace(packageDirectory, workspace);
    }

    public McpClientProcess newClient() throws IOException {
        if (!Files.isRegularFile(serviceJar)) {
            throw new IOException("MCP executable is missing: " + serviceJar);
        }
        Path whitelist = packageDirectory.resolve("config/weka-whitelist.yaml");
        Path searchSpace = packageDirectory.resolve("config/automl-search.yaml");
        if (!Files.isRegularFile(whitelist) || !Files.isRegularFile(searchSpace)) {
            throw new IOException("MCP configuration resources are missing from " + packageDirectory.resolve("config"));
        }
        Files.createDirectories(workspaceDirectory.resolve("experiments"));
        Properties properties = new Properties();
        Path defaults = packageDirectory.resolve("config/config.properties");
        if (Files.isRegularFile(defaults)) {
            try (var reader = Files.newBufferedReader(defaults, java.nio.charset.StandardCharsets.UTF_8)) {
                properties.load(reader);
            }
        }
        properties.setProperty("dataRoot", portable(dataDirectory));
        properties.setProperty("experimentRoot", portable(workspaceDirectory.resolve("experiments")));
        properties.setProperty("whitelistPath", portable(whitelist));
        properties.setProperty("searchSpacePath", portable(searchSpace));
        try (var writer = Files.newBufferedWriter(configFile, java.nio.charset.StandardCharsets.UTF_8)) {
            properties.store(writer, "Weka MCP GUI workspace");
        }
        return new McpClientProcess(serviceJar, configFile);
    }

    public String importDataset(Path source) throws IOException {
        Path input = source.toRealPath();
        if (!Files.isRegularFile(input)) {
            throw new IllegalArgumentException("Dataset must be a regular file");
        }
        String name = input.getFileName().toString().toLowerCase(Locale.ROOT);
        if (!name.endsWith(".arff") && !name.endsWith(".csv")) {
            throw new IllegalArgumentException("Dataset must be an ARFF or CSV file");
        }
        if (Files.size(input) > 104_857_600L) {
            throw new IllegalArgumentException("Dataset exceeds the 100 MB input limit");
        }
        if (input.startsWith(dataDirectory)) {
            return dataDirectory.relativize(input).toString().replace('\\', '/');
        }
        String targetName = UUID.randomUUID() + "-" + safeFilename(input.getFileName().toString());
        Path target = dataDirectory.resolve(targetName);
        Path temporary = Files.createTempFile(dataDirectory, ".upload-", ".tmp");
        try {
            Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException error) {
            Files.deleteIfExists(temporary);
            throw error;
        }
        return targetName;
    }

    public String writeInstances(String filename, Instances data) throws IOException {
        if (!filename.toLowerCase(Locale.ROOT).endsWith(".arff")) {
            throw new IllegalArgumentException("Explorer instances must be exported as ARFF");
        }
        Path target = dataDirectory.resolve(UUID.randomUUID() + "-" + safeFilename(filename));
        Path temporary = Files.createTempFile(dataDirectory, ".instances-", ".tmp");
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(temporary.toFile());
            saver.writeBatch();
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (Exception error) {
            Files.deleteIfExists(temporary);
            throw error;
        }
        return target.getFileName().toString();
    }

    public Path resolveDataset(String relative) throws IOException {
        Path candidate = dataDirectory.resolve(relative).normalize();
        if (!candidate.startsWith(dataDirectory)) {
            throw new IllegalArgumentException("Dataset must be inside the plugin data directory: "
                    + candidate + "; root=" + dataDirectory + "; relative=" + relative);
        }
        Path real = candidate.toRealPath();
        if (!real.startsWith(dataDirectory) || !Files.isRegularFile(real)) {
            throw new IllegalArgumentException("Dataset must be a regular file inside the plugin data directory");
        }
        return real;
    }

    public Path workspaceDirectory() {
        return workspaceDirectory;
    }

    public Path packageDirectory() {
        return packageDirectory;
    }

    public Path serviceJar() {
        return serviceJar;
    }

    public Path configFile() {
        return configFile;
    }

    private static String safeFilename(String value) {
        String safe = value.replaceAll("[^\\p{L}\\p{N}._-]", "_");
        return safe.substring(0, Math.min(100, safe.length()));
    }

    private static String portable(Path path) {
        return path.toAbsolutePath().normalize().toString().replace('\\', '/');
    }
}
