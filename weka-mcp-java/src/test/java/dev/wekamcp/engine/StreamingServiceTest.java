package dev.wekamcp.engine;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import io.modelcontextprotocol.spec.McpSchema;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StreamingServiceTest {
    @TempDir
    Path directory;

    @Test
    void incrementalAlgorithmsTrainPersistAndRestore() throws Exception {
        try (Fixture fixture = new Fixture()) {
            for (String classifier : List.of(
                    "weka.classifiers.bayes.NaiveBayesUpdateable", "weka.classifiers.lazy.IBk",
                    "weka.classifiers.bayes.NaiveBayesMultinomialUpdateable", "weka.classifiers.trees.HoeffdingTree")) {
                JsonNode trained = success(fixture.tools.trainStream(Map.of(
                        "path", "iris.arff", "classifier_class", classifier, "class_index", 5)));
                assertStream(trained, classifier);
                assertEquals(30, trained.get("read_instances").intValue());
                assertEquals(24, trained.get("training_instances").intValue());
                assertEquals(6, trained.get("test_instances").intValue());
                String experimentId = trained.get("experiment_id").asText();
                JsonNode manifest = success(fixture.tools.getExperiment(Map.of("experiment_id", experimentId)));
                assertEquals("iris.arff", manifest.get("sourcePath").asText());
                assertEquals("stream", manifest.get("protocol").asText());
                assertFalse(Files.exists(directory.resolve("experiments").resolve(experimentId).resolve("dataset.arff")));
                JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of(
                        "experiment_id", trained.get("experiment_id").asText())));
                String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                        "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
                assertEquals(success(fixture.tools.predict(Map.of(
                                "model_id", trained.get("model_id").asText(), "instance_json", instance))).get("predictions"),
                        success(fixture.tools.predict(Map.of(
                                "model_id", restored.get("model_id").asText(), "instance_json", instance))).get("predictions"));
            }
            Files.writeString(directory.resolve("binary.arff"), """
                    @relation binary
                    @attribute x numeric
                    @attribute y numeric
                    @attribute class {a,b}
                    @data
                    0,0,a
                    0.1,0.2,a
                    0.2,0.1,a
                    0.3,0.2,a
                    0.4,0.3,a
                    1,1,b
                    1.1,1.2,b
                    1.2,1.1,b
                    1.3,1.2,b
                    1.4,1.3,b
                    """);
            JsonNode sgd = success(fixture.tools.trainStream(Map.of(
                    "path", "binary.arff", "classifier_class", "weka.classifiers.functions.SGD",
                    "options", List.of("-F", "1", "-L", "0.01", "-S", "42"))));
            assertStream(sgd, "weka.classifiers.functions.SGD");
            assertEquals(8, sgd.get("training_instances").intValue());
            assertEquals(2, sgd.get("test_instances").intValue());
            assertEquals(10, sgd.get("read_instances").intValue());
            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of(
                    "experiment_id", sgd.get("experiment_id").asText())));
            Map<String, Object> input = Map.of("x", 0.0, "y", 0.0);
            String instance = ToolSupport.JSON.writeValueAsString(input);
            assertEquals(success(fixture.tools.predict(Map.of(
                            "model_id", sgd.get("model_id").asText(), "instance_json", instance))).get("predictions"),
                    success(fixture.tools.predict(Map.of(
                            "model_id", restored.get("model_id").asText(), "instance_json", instance))).get("predictions"));
        }
    }

    @Test
    void csvSupportsPrequentialLimitsAndRejectsBatchClassifiers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            Files.writeString(directory.resolve("binary.csv"), """
                    x,y,class
                    0,0,a
                    0.1,0.2,a
                    0.2,0.1,a
                    0.3,0.2,a
                    0.4,0.3,a
                    1,1,b
                    1.1,1.2,b
                    1.2,1.1,b
                    1.3,1.2,b
                    1.4,1.3,b
                    """);
            JsonNode prequential = success(fixture.tools.trainStream(Map.of(
                    "path", "binary.csv", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable",
                    "evaluation", "prequential", "test_every", 1, "max_test_instances", 3, "persist", false)));
            assertEquals(10, prequential.get("read_instances").intValue());
            assertEquals(10, prequential.get("training_instances").intValue());
            assertEquals(3, prequential.get("test_instances").intValue());
            assertTrue(prequential.get("evaluation").get("test_then_train").booleanValue());
            assertFalse(prequential.has("experiment_id"));
            assertEquals(0, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());

            JsonNode bounded = success(fixture.tools.trainStream(Map.of(
                    "path", "binary.csv", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable",
                    "max_instances", 4)));
            assertEquals(4, bounded.get("read_instances").intValue());
            assertEquals("max_instances", bounded.get("stop_reason").asText());
            assertTrue(bounded.get("partial").booleanValue());
            assertTrue(bounded.get("evaluation").isNull());
            assertTrue(fixture.tools.trainStream(Map.of(
                    "path", "iris.arff", "classifier_class", "weka.classifiers.trees.J48")).isError());
        }
    }

    private static void assertStream(JsonNode trained, String classifier) {
        assertEquals(classifier, trained.get("classifier_class").asText());
        assertEquals("TRAINED", trained.get("model_status").asText());
        assertEquals("end_of_file", trained.get("stop_reason").asText());
        assertFalse(trained.get("partial").booleanValue());
        assertNotNull(trained.get("evaluation").get("accuracy_percent"));
        assertEquals("holdout", trained.get("stream").get("evaluation").asText());
    }

    private static JsonNode success(CallToolResult result) {
        assertFalse(result.isError(), () -> text(result));
        return ToolSupport.JSON.readTree(text(result));
    }

    private static String text(CallToolResult result) {
        return ((McpSchema.TextContent) result.content().get(0)).text();
    }

    private final class Fixture implements AutoCloseable {
        private final TaskRunner tasks;
        private final WekaTools tools;
        private final ServerConfig config;

        private Fixture() throws Exception {
            try (InputStream input = StreamingServiceTest.class.getResourceAsStream("/iris.arff")) {
                assertNotNull(input);
                Files.copy(input, directory.resolve("iris.arff"), StandardCopyOption.REPLACE_EXISTING);
            }
            Path configPath = directory.resolve("config.properties");
            Files.writeString(configPath,
                    "dataRoot=" + directory.toString().replace('\\', '/') + "\n"
                            + "whitelistPath=" + Path.of("weka-whitelist.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n"
                            + "searchSpacePath=" + Path.of("automl-search.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n");
            config = ServerConfig.load(configPath);
            Whitelist whitelist = Whitelist.load(config.whitelistPath());
            whitelist.selfCheck();
            tasks = new TaskRunner(config);
            tools = new WekaTools(config, whitelist, new SessionRegistry(config), tasks);
        }

        @Override
        public void close() {
            tasks.close();
        }
    }
}
