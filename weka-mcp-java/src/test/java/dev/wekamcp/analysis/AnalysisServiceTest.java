package dev.wekamcp.analysis;

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

class AnalysisServiceTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void initializeWeka() {
        WekaPackageManager.loadPackages(false, true, false);
    }

    @Test
    void profileDatasetReportsUsefulStatistics() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            JsonNode profile = success(fixture.tools.profileDataset(Map.of("dataset_id", datasetId)));
            assertEquals(30, profile.get("instances").intValue());
            assertEquals(5, profile.get("attributes").size());
            assertEquals(0, profile.get("duplicate_rows").intValue());
            assertEquals(0, profile.get("missing_cells").intValue());
            assertEquals(3, profile.get("class_distribution").size());
            assertEquals(1.0, profile.get("class_imbalance_ratio").doubleValue());
            assertEquals(5, profile.get("attributes").size());
            assertNotNull(profile.get("attributes").get(0).get("mean"));
            assertNotNull(profile.get("attributes").get(4).get("nominal_counts"));
            assertFalse(profile.get("correlations").isEmpty());
        }
    }

    @Test
    void learningCurveUsesAFixedOuterTestSet() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode curve = success(fixture.tools.learningCurve(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48",
                    "fractions", "0.25,0.5,1.0")));
            assertEquals(3, curve.get("curves").size());
            assertEquals(6, curve.get("curves").get(0).get("test_instances").intValue());
            assertEquals(6, curve.get("curves").get(1).get("test_instances").intValue());
            assertEquals(6, curve.get("curves").get(2).get("test_instances").intValue());
            assertTrue(curve.get("curves").get(0).get("training_instances").intValue()
                    < curve.get("curves").get(1).get("training_instances").intValue());
            assertTrue(curve.get("curves").get(1).get("training_instances").intValue()
                    < curve.get("curves").get(2).get("training_instances").intValue());
            assertNotNull(curve.get("curves").get(2).get("accuracy_percent"));
        }
    }

    @Test
    void diagnosticsCoverClassificationAndRegression() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            JsonNode diagnostics = success(fixture.tools.modelDiagnostics(Map.of(
                    "model_id", modelId, "dataset_id", datasetId)));
            assertEquals(3, diagnostics.get("curves").size());
            assertNotNull(diagnostics.get("calibration"));
            assertNotNull(diagnostics.get("errors"));
            assertNotNull(diagnostics.get("accuracy_percent"));

            String regressionId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff")))
                    .get("dataset_id").asText();
            String regressionModel = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", regressionId, "classifier_class", "weka.classifiers.rules.ZeroR")))
                    .get("model_id").asText();
            JsonNode residuals = success(fixture.tools.modelDiagnostics(Map.of(
                    "model_id", regressionModel, "dataset_id", regressionId)));
            assertNotNull(residuals.get("mean_residual"));
            assertNotNull(residuals.get("rmse"));
            assertNotNull(residuals.get("residual_histogram"));
            assertNotNull(residuals.get("largest_errors"));
        }
    }

    @Test
    void selectsAttributesMinesRulesAndExplainsModels() throws Exception {
        try (Fixture fixture = new Fixture(directory)) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode infoGain = success(fixture.tools.selectAttributes(Map.of(
                    "dataset_id", datasetId, "method", "info_gain", "max_attributes", 2)));
            assertEquals(2, infoGain.get("selected_attributes").size());
            assertEquals(2, infoGain.get("ranking").size());
            JsonNode cfs = success(fixture.tools.selectAttributes(Map.of(
                    "dataset_id", datasetId, "method", "cfs", "max_attributes", 2)));
            assertEquals(2, cfs.get("selected_attributes").size());

            String weatherId = success(fixture.tools.loadDataset(Map.of("path", "weather.nominal.arff")))
                    .get("dataset_id").asText();
            JsonNode rules = success(fixture.tools.associate(Map.of(
                    "dataset_id", weatherId, "min_support", 0.1, "min_confidence", 0.1, "max_rules", 5)));
            assertTrue(rules.get("rule_count").intValue() > 0);
            assertTrue(rules.get("rule_count").intValue() <= 5);
            assertNotNull(rules.get("rules").get(0).get("confidence"));

            String treeId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            JsonNode explanation = success(fixture.tools.explainModel(Map.of("model_id", treeId)));
            assertTrue(explanation.get("supported").booleanValue());
            assertNotNull(explanation.get("tree"));
            assertNotNull(explanation.get("dot"));

            String forestId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.RandomForest")))
                    .get("model_id").asText();
            JsonNode unsupported = success(fixture.tools.explainModel(Map.of("model_id", forestId)));
            assertFalse(unsupported.get("supported").booleanValue());
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
            Path data = Files.createDirectories(directory.resolve("data"));
            for (String name : List.of("iris.arff", "regression.arff", "weather.nominal.arff")) {
                try (InputStream input = AnalysisServiceTest.class.getResourceAsStream("/" + name)) {
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

        @Override
        public void close() {
            tasks.close();
        }
    }
}
