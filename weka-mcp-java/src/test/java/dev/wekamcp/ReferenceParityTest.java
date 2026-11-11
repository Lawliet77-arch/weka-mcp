package dev.wekamcp;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import io.modelcontextprotocol.spec.McpSchema;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class ReferenceParityTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void initializeWeka() {
        WekaPackageManager.loadPackages(false, true, false);
    }

    @Test
    void nominalClassifiersMatchDirectWekaPredictions() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            Instances data = fixture.load("iris.arff");
            data.setClassIndex(4);
            Instance instance = new DenseInstance(4);
            instance.setDataset(data);
            double[] values = {5.1, 3.5, 1.4, 0.2};
            for (int index = 0; index < values.length; index++) {
                instance.setValue(index, values[index]);
            }
            String instanceJson = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            for (String className : List.of(
                    "weka.classifiers.trees.J48",
                    "weka.classifiers.trees.RandomForest",
                    "weka.classifiers.bayes.NaiveBayes",
                    "weka.classifiers.functions.Logistic")) {
                Classifier direct = (Classifier) Utils.forName(Classifier.class, className, new String[0]);
                direct.buildClassifier(data);
                double[] expected = direct.distributionForInstance(instance);
                String modelId = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", datasetId, "classifier_class", className)))
                        .get("model_id").asText();
                JsonNode prediction = success(fixture.tools.predict(Map.of(
                        "model_id", modelId, "instance_json", instanceJson)))
                        .get("predictions").get(0);
                assertEquals(data.classAttribute().value(Utils.maxIndex(expected)),
                        prediction.get("predicted_value").asText(), className);
                for (int index = 0; index < expected.length; index++) {
                    assertEquals(expected[index], prediction.get("class_distribution")
                            .get(data.classAttribute().value(index)).doubleValue(), 1e-9, className);
                }
            }
        }
    }

    @Test
    void regressionModelsMatchDirectWekaPredictions() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff", "class_index", 2)))
                    .get("dataset_id").asText();
            Instances data = fixture.load("regression.arff");
            data.setClassIndex(1);
            Instance instance = new DenseInstance(2);
            instance.setDataset(data);
            instance.setValue(0, 5.5);
            instance.setValue(1, 0);
            String instanceJson = ToolSupport.JSON.writeValueAsString(Map.of("x", 5.5));
            for (String className : List.of(
                    "weka.classifiers.functions.LinearRegression",
                    "weka.classifiers.functions.SMOreg",
                    "weka.classifiers.trees.RandomForest")) {
                Classifier direct = (Classifier) Utils.forName(Classifier.class, className, new String[0]);
                direct.buildClassifier(data);
                double expected = direct.classifyInstance(instance);
                String modelId = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", datasetId, "classifier_class", className)))
                        .get("model_id").asText();
                JsonNode prediction = success(fixture.tools.predict(Map.of(
                        "model_id", modelId, "instance_json", instanceJson)))
                        .get("predictions").get(0);
                assertEquals(expected, prediction.get("predicted_value").doubleValue(), 1e-9, className);
            }
        }
    }

    @Test
    void kMeansAssignmentsMatchDirectWeka() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            Instances data = fixture.load("iris.arff");
            Remove remove = new Remove();
            remove.setAttributeIndices("5");
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            data.setClassIndex(-1);
            SimpleKMeans direct = new SimpleKMeans();
            direct.setNumClusters(3);
            direct.setOptions(new String[]{"-S", "42"});
            direct.buildClusterer(data);
            JsonNode clustered = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId,
                    "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3,
                    "options", List.of("-S", "42"))));
            for (int index = 0; index < data.numInstances(); index++) {
                assertEquals(direct.clusterInstance(data.instance(index)),
                        clustered.get("assignments").get(index).intValue());
            }
        }
    }

    private static JsonNode success(CallToolResult result) {
        assertFalse(result.isError(), () -> text(result));
        return ToolSupport.JSON.readTree(text(result));
    }

    private static String text(CallToolResult result) {
        return ((McpSchema.TextContent) result.content().get(0)).text();
    }

    private static final class Fixture implements AutoCloseable {
        private final TaskRunner tasks;
        private final WekaTools tools;
        private final Path root;

        private Fixture(Path directory) throws Exception {
            this.root = directory;
            Path data = Files.createDirectories(directory.resolve("data"));
            for (String name : List.of("iris.arff", "regression.arff")) {
                try (InputStream input = ReferenceParityTest.class.getResourceAsStream("/" + name)) {
                    Files.copy(input, data.resolve(name), StandardCopyOption.REPLACE_EXISTING);
                }
            }
            Path configPath = directory.resolve("config.properties");
            Files.writeString(configPath,
                    "dataRoot=" + data.toString().replace('\\', '/') + "\n"
                            + "experimentRoot=" + directory.resolve("experiments").toString().replace('\\', '/') + "\n"
                            + "whitelistPath=" + Path.of("weka-whitelist.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n"
                            + "searchSpacePath=" + Path.of("automl-search.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n");
            ServerConfig config = ServerConfig.load(configPath);
            Whitelist whitelist = Whitelist.load(config.whitelistPath());
            whitelist.selfCheck();
            tasks = new TaskRunner(config);
            tools = new WekaTools(config, whitelist, new SessionRegistry(config), tasks);
        }

        private Instances load(String name) throws Exception {
            return dev.wekamcp.engine.DatasetIo.load(root.resolve("data").resolve(name), "auto");
        }

        @Override
        public void close() {
            tasks.close();
        }
    }
}
