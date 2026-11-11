package dev.wekamcp.automl;

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
import weka.core.WekaPackageManager;

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

class AutoMlServiceTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void initializeWeka() {
        WekaPackageManager.loadPackages(false, true, false);
    }

    @Test
    void nestedCvReturnsIndependentMetricsAndRestoresExperiment() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            JsonNode automatic = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId,
                    "protocol", "nested_cv",
                    "outer_folds", 3,
                    "inner_folds", 2,
                    "search", "grid",
                    "maxCandidates", 4)));
            assertEquals("nested_cv", automatic.get("protocol").asText());
            assertEquals(4, automatic.get("candidate_count").intValue());
            assertEquals(4, automatic.get("candidates").size());
            assertNotNull(automatic.get("validation").get("mean_validation_score"));
            assertNotNull(automatic.get("evaluation").get("accuracy_percent"));
            assertNotNull(automatic.get("test_score"));
            assertEquals(automatic.get("selection_score").doubleValue(),
                    automatic.get("validation_score").doubleValue());

            String experimentId = automatic.get("experiment_id").asText();
            JsonNode manifest = success(fixture.tools.getExperiment(Map.of("experiment_id", experimentId)));
            assertEquals("nested_cv", manifest.get("protocol").asText());
            assertEquals(1, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());

            JsonNode holdout = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId, "maxCandidates", 2)));
            String secondId = holdout.get("experiment_id").asText();
            JsonNode comparison = success(fixture.tools.compareExperiments(Map.of(
                    "experiment_ids", List.of(experimentId, secondId))));
            assertEquals(2, comparison.get("experiments").size());

            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of("experiment_id", experimentId)));
            assertEquals(experimentId, restored.get("experiment_id").asText());
            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode prediction = success(fixture.tools.predict(Map.of(
                    "model_id", restored.get("model_id").asText(), "instance_json", instance)));
            assertNotNull(prediction.get("predictions").get(0).get("predicted_value"));

            assertFalse(fixture.tools.deleteExperiment(Map.of("experiment_id", experimentId)).isError());
            assertFalse(fixture.tools.deleteExperiment(Map.of("experiment_id", secondId)).isError());
            assertEquals(0, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());
        }
    }

    @Test
    void randomSearchIsDeterministicAndHonorsCandidateCap() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            Map<String, Object> arguments = Map.of(
                    "dataset_id", datasetId,
                    "search", "random",
                    "maxCandidates", 4,
                    "seed", 42L,
                    "protocol", "legacy_cv");
            JsonNode first = success(fixture.tools.autoClassify(arguments));
            JsonNode second = success(fixture.tools.autoClassify(arguments));
            assertEquals(4, first.get("candidate_count").intValue());
            assertEquals("weka.classifiers.rules.ZeroR",
                    first.get("candidates").get(0).get("classifier_class").asText());
            assertEquals(first.get("classifier_class").asText(), second.get("classifier_class").asText());
            assertEquals(first.get("candidates"), second.get("candidates"));
        }
    }

    @Test
    void exhaustedBudgetReturnsPartialExperiment() throws Exception {
        try (Fixture fixture = new Fixture(directory, true)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode result = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId,
                    "protocol", "nested_cv",
                    "outer_folds", 30,
                    "inner_folds", 2,
                    "search", "grid",
                    "maxCandidates", 12,
                    "budgetSeconds", 1)));
            assertTrue(result.get("partial").booleanValue());
            assertTrue(result.get("completed_candidates").intValue() < result.get("candidate_count").intValue());
            assertEquals("NOT_TRAINED", result.get("final_model_status").asText());
            assertTrue(result.get("model_id").isNull());
            assertTrue(result.get("experiment_id").isNull());
            assertEquals(0, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());
        }
    }

    @Test
    void successiveHalvingEliminatesCandidatesAndPersistsStages() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            Map<String, Object> arguments = Map.of(
                    "dataset_id", datasetId, "search", "halving", "maxCandidates", 6, "folds", 2,
                    "seed", 42L, "budgetSeconds", 120);
            JsonNode result = success(fixture.tools.autoClassify(arguments));
            assertEquals("halving", result.get("search").asText());
            assertEquals("TRAINED", result.get("final_model_status").asText());
            assertEquals(3, result.get("halving_stages").size());
            assertEquals(6, result.get("halving_stages").get(0).get("completed_candidates").intValue());
            assertEquals(3, result.get("halving_stages").get(0).get("retained_candidates").size());
            assertEquals(3, result.get("halving_stages").get(0).get("eliminated_candidates").size());
            assertEquals(1, result.get("halving_stages").get(1).get("eliminated_candidates").size());
            assertEquals(2, result.get("halving_stages").get(2).get("completed_candidates").intValue());
            assertEquals(1.0, result.get("halving_stages").get(2).get("resource_fraction").doubleValue());
            assertFalse(result.get("partial").booleanValue());
            assertNotNull(result.get("test_score"));
            JsonNode manifest = success(fixture.tools.getExperiment(Map.of(
                    "experiment_id", result.get("experiment_id").asText())));
            assertEquals(result.get("halving_stages"), manifest.get("search").get("stages"));
            JsonNode copied = success(fixture.tools.saveModel(Map.of("model_id", result.get("model_id").asText())));
            JsonNode copiedManifest = success(fixture.tools.getExperiment(Map.of(
                    "experiment_id", copied.get("experiment_id").asText())));
            assertEquals(manifest.get("search"), copiedManifest.get("search"));
            assertEquals(manifest.get("candidates"), copiedManifest.get("candidates"));
            assertEquals(manifest.get("id"), copiedManifest.get("parentExperimentId"));

            JsonNode repeated = success(fixture.tools.autoClassify(arguments));
            assertEquals(result.get("classifier_class"), repeated.get("classifier_class"));
            assertEquals(result.get("halving_stages"), repeated.get("halving_stages"));
            assertEquals(result.get("candidates"), repeated.get("candidates"));
        }
    }

    @Test
    void halvingBudgetNeverPersistsAnIncompleteFinalModel() throws Exception {
        try (Fixture fixture = new Fixture(directory, true, 2)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode result = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId, "search", "halving", "protocol", "legacy_cv",
                    "maxCandidates", 6, "folds", 2, "budgetSeconds", 1)));
            assertTrue(result.get("partial").booleanValue());
            assertTrue(result.get("completed_candidates").intValue() < result.get("candidate_count").intValue());
            assertEquals("NOT_TRAINED", result.get("final_model_status").asText());
            assertTrue(result.get("model_id").isNull());
            assertTrue(result.get("experiment_id").isNull());
            assertEquals(0, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());
        }
    }

    @Test
    void nestedHalvingUsesOuterTrainingOnlyAndRestoresModel() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            Map<String, Object> arguments = Map.of(
                    "dataset_id", datasetId, "search", "halving", "protocol", "nested_cv",
                    "outer_folds", 3, "inner_folds", 2, "maxCandidates", 4, "seed", 42L, "budgetSeconds", 120);
            JsonNode result = success(fixture.tools.autoClassify(arguments));
            assertEquals("TRAINED", result.get("final_model_status").asText());
            assertEquals(3, result.get("nested_outer_folds").size());
            assertTrue(result.get("nested_cv_complete").booleanValue());
            assertFalse(result.get("partial").booleanValue());
            assertEquals(3, result.get("evaluation").get("outer_folds").intValue());
            assertEquals(3, result.get("evaluation").get("completed_outer_folds").intValue());
            assertEquals(2, result.get("evaluation").get("inner_folds").intValue());
            assertEquals("outer_training_only", result.get("evaluation").get("selection_scope").asText());
            assertTrue(result.get("evaluation").get("nested_cv_complete").booleanValue());
            assertNotNull(result.get("test_score"));
            assertTrue(result.get("candidates").findValues("scope").stream()
                    .anyMatch(value -> "outer_training_fold".equals(value.asText())));
            for (JsonNode winner : result.get("nested_outer_folds")) {
                assertFalse(winner.get("classifier_class").asText().isBlank());
                assertEquals(3, winner.get("completed_stages").intValue());
            }
            JsonNode manifest = success(fixture.tools.getExperiment(Map.of(
                    "experiment_id", result.get("experiment_id").asText())));
            assertEquals(3, manifest.get("search").get("outer_folds").size());
            assertTrue(manifest.get("search").get("nested_cv_complete").booleanValue());
            JsonNode repeated = success(fixture.tools.autoClassify(arguments));
            assertEquals(result.get("nested_outer_folds"), repeated.get("nested_outer_folds"));
            assertEquals(result.get("evaluation"), repeated.get("evaluation"));

            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode expected = success(fixture.tools.predict(Map.of(
                    "model_id", result.get("model_id").asText(), "instance_json", instance))).get("predictions");
            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of(
                    "experiment_id", result.get("experiment_id").asText())));
            assertEquals(expected, success(fixture.tools.predict(Map.of(
                    "model_id", restored.get("model_id").asText(), "instance_json", instance))).get("predictions"));
        }
    }

    @Test
    void nestedHalvingBudgetReturnsExplicitIncompleteState() throws Exception {
        try (Fixture fixture = new Fixture(directory, true, 2)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode result = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId, "search", "halving", "protocol", "nested_cv",
                    "outer_folds", 3, "inner_folds", 2, "maxCandidates", 4, "budgetSeconds", 1)));
            assertTrue(result.get("partial").booleanValue());
            assertFalse(result.get("nested_cv_complete").booleanValue());
            assertEquals("outer_training_fold", result.get("selection_source").asText());
            assertEquals("NOT_TRAINED", result.get("final_model_status").asText());
            assertTrue(result.get("model_id").isNull());
            assertTrue(result.get("experiment_id").isNull());
            assertEquals(0, success(fixture.tools.listExperiments(Map.of())).get("total").intValue());
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

        private Fixture(Path directory) throws Exception {
            this(directory, false);
        }

        private Fixture(Path directory, boolean delayed) throws Exception {
            this(directory, delayed, 30);
        }

        private Fixture(Path directory, boolean delayed, int delayAfter) throws Exception {
            Path data = Files.createDirectories(directory.resolve("data"));
            try (InputStream input = AutoMlServiceTest.class.getResourceAsStream("/iris.arff")) {
                Files.copy(input, data.resolve("iris.arff"), StandardCopyOption.REPLACE_EXISTING);
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
            tasks = delayed ? new SlowTaskRunner(config, delayAfter) : new TaskRunner(config);
            tools = new WekaTools(config, whitelist, new SessionRegistry(config), tasks);
        }

        @Override
        public void close() {
            tasks.close();
        }

        private static final class SlowTaskRunner extends TaskRunner {
            private final int delayAfter;
            private int calls;

            private SlowTaskRunner(ServerConfig config, int delayAfter) {
                super(config);
                this.delayAfter = delayAfter;
            }

            @Override
            public <T> T call(long timeoutSeconds, java.util.concurrent.Callable<T> task) throws Exception {
                if (++calls > delayAfter) {
                    Thread.sleep(1100);
                }
                return super.call(timeoutSeconds, task);
            }
        }
    }
}
