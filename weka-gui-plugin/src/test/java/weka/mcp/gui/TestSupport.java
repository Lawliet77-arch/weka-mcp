package weka.mcp.gui;

import weka.core.Instances;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

final class TestSupport {
    private TestSupport() {
    }

    static PluginWorkspace workspace() throws IOException {
        Path packageDirectory = Path.of("target/test-package");
        Path service = Path.of("../weka-mcp-java/target/weka-mcp-java-1.6.0.jar")
                .toAbsolutePath().normalize();
        if (!Files.isRegularFile(service)) {
            throw new IOException("Build the MCP service before running GUI tests: " + service);
        }
        Files.createDirectories(packageDirectory.resolve("tools"));
        Files.createDirectories(packageDirectory.resolve("config"));
        Files.copy(service, packageDirectory.resolve("tools/weka-mcp-java-1.6.0.jar"),
                StandardCopyOption.REPLACE_EXISTING);
        for (String name : List.of("weka-whitelist.yaml", "automl-search.yaml", "config.properties")) {
            Files.copy(Path.of("../weka-mcp-java").resolve(name), packageDirectory.resolve("config").resolve(name),
                    StandardCopyOption.REPLACE_EXISTING);
        }
        return new PluginWorkspace(packageDirectory, Path.of("target/test-workspace"));
    }

    static Instances iris() throws Exception {
        Path data = Path.of("target/test-workspace/data");
        Files.createDirectories(data);
        Path iris = data.resolve("iris.arff");
        try (InputStream input = TestSupport.class.getResourceAsStream("/iris.arff")) {
            if (input == null) {
                throw new IOException("Missing test resource iris.arff");
            }
            Files.copy(input, iris, StandardCopyOption.REPLACE_EXISTING);
        }
        Instances instances = ExplorerDataOperations.loadDataset(iris);
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    static Instances weather() throws Exception {
        Path data = Path.of("target/test-workspace/data");
        Files.createDirectories(data);
        Path weather = data.resolve("weather.arff");
        try (InputStream input = TestSupport.class.getResourceAsStream("/weather.arff")) {
            if (input == null) {
                throw new IOException("Missing test resource weather.arff");
            }
            Files.copy(input, weather, StandardCopyOption.REPLACE_EXISTING);
        }
        Instances instances = ExplorerDataOperations.loadDataset(weather);
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    static void writeUpdate() throws IOException {
        Files.writeString(Path.of("target/test-workspace/data/update.arff"), """
                @relation iris
                @attribute sepallength numeric
                @attribute sepalwidth numeric
                @attribute petallength numeric
                @attribute petalwidth numeric
                @attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}
                @data
                5.1,3.5,1.4,0.2,Iris-setosa
                6.2,3.4,5.4,2.3,Iris-virginica
                """);
    }
}
