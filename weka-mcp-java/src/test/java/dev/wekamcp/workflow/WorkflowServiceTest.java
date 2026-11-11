package dev.wekamcp.workflow;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
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

class WorkflowServiceTest {
    @TempDir
    Path directory;

    @Test
    void splitPreviewHistogramAndScatterSupportAnalysis() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            Map<String, Object> arguments = Map.of("dataset_id", datasetId, "train_percent", 80, "seed", 42);
            JsonNode first = success(fixture.tools.splitDataset(arguments));
            JsonNode second = success(fixture.tools.splitDataset(arguments));
            assertEquals(24, first.get("training_instances").intValue());
            assertEquals(6, first.get("test_instances").intValue());
            assertTrue(first.get("stratified").booleanValue());
            assertEquals(first.get("training_instances"), second.get("training_instances"));
            assertEquals(first.get("test_instances"), second.get("test_instances"));

            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", first.get("training_dataset_id").asText(),
                    "classifier_class", "weka.classifiers.trees.J48"))).get("model_id").asText();
            JsonNode evaluated = success(fixture.tools.evaluateModel(Map.of(
                    "model_id", modelId, "test_dataset_id", first.get("test_dataset_id").asText())));
            assertEquals(6, evaluated.get("instances").intValue());
            assertTrue(evaluated.get("independent_test").booleanValue());

            JsonNode preview = success(fixture.tools.transformPreview(Map.of(
                    "dataset_id", datasetId, "filter_class", "weka.filters.unsupervised.attribute.Normalize",
                    "max_rows", 2)));
            assertEquals(2, preview.get("rows").size());
            assertFalse(preview.get("fold_safe").booleanValue());
            assertEquals("full_dataset", preview.get("fit_scope").asText());
            assertTrue(preview.get("rows").get(0).get(0).isNumber());
            JsonNode discretized = success(fixture.tools.transformPreview(Map.of(
                    "dataset_id", datasetId, "filter_class", "weka.filters.supervised.attribute.Discretize",
                    "class_index", 5, "max_rows", 1)));
            assertTrue(discretized.get("rows").get(0).get(0).isTextual());
            assertEquals(0, success(fixture.tools.transformPreview(Map.of(
                    "dataset_id", datasetId, "filter_class", "weka.filters.unsupervised.attribute.Normalize",
                    "max_rows", 0))).get("rows").size());

            JsonNode numeric = success(fixture.tools.histogram(Map.of(
                    "dataset_id", datasetId, "attribute_index", 1, "bins", 4)));
            assertEquals(4, numeric.get("bins").size());
            assertEquals(30, numeric.get("bins").findValues("count").stream().mapToInt(JsonNode::intValue).sum());
            JsonNode nominal = success(fixture.tools.histogram(Map.of(
                    "dataset_id", datasetId, "attribute_index", 5)));
            assertEquals(3, nominal.get("bins").size());
            for (JsonNode bin : nominal.get("bins")) {
                assertEquals(10, bin.get("count").intValue());
            }
            JsonNode scatter = success(fixture.tools.scatter(Map.of(
                    "dataset_id", datasetId, "x_index", 1, "y_index", 3, "max_points", 10)));
            assertEquals(10, scatter.get("count").intValue());
            assertTrue(scatter.get("sampled").booleanValue());
            assertNotNull(scatter.get("points").get(0).get("class"));
            JsonNode drift = success(fixture.tools.compareDatasets(Map.of(
                    "dataset_id", datasetId, "other_dataset_id", first.get("test_dataset_id").asText())));
            assertTrue(drift.get("same_header").booleanValue());
            assertFalse(drift.get("attributes").isEmpty());
            assertNotNull(drift.get("attributes").get(0).get("psi"));
        }
    }

    @Test
    void exportsRoundTripAndRejectsInvalidPaths() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            for (String format : List.of("arff", "csv")) {
                JsonNode exported = success(fixture.tools.exportDataset(Map.of(
                        "dataset_id", datasetId, "path", "exports/iris." + format, "format", format)));
                assertEquals(30, exported.get("instances").intValue());
                assertTrue(exported.get("bytes").longValue() > 0);
                assertEquals(64, exported.get("sha256").asText().length());
                JsonNode loaded = success(fixture.tools.loadDataset(Map.of(
                        "path", "exports/iris." + format, "format", format, "class_index", 5)));
                assertEquals(30, loaded.get("instances").intValue());
                assertEquals(5, loaded.get("attributes").intValue());
            }
            assertTrue(fixture.tools.exportDataset(Map.of(
                    "dataset_id", datasetId, "path", "exports/iris.arff")).isError());
            assertTrue(fixture.tools.exportDataset(Map.of(
                    "dataset_id", datasetId, "path", "../escape.arff")).isError());
            String unlabeledId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            assertTrue(fixture.tools.transformPreview(Map.of(
                    "dataset_id", unlabeledId, "filter_class", "weka.filters.supervised.attribute.Discretize")).isError());
        }
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

        private Fixture() throws Exception {
            try (InputStream input = WorkflowServiceTest.class.getResourceAsStream("/iris.arff")) {
                assertNotNull(input);
                Files.copy(input, directory.resolve("iris.arff"), StandardCopyOption.REPLACE_EXISTING);
            }
            Path configPath = directory.resolve("config.properties");
            Files.writeString(configPath,
                    "dataRoot=" + directory.toString().replace('\\', '/') + "\n"
                            + "whitelistPath=" + Path.of("weka-whitelist.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n"
                            + "searchSpacePath=" + Path.of("automl-search.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n");
            ServerConfig config = ServerConfig.load(configPath);
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
