package dev.wekamcp.evaluation;

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

class ModelComparisonServiceTest {
    @TempDir
    Path directory;

    @Test
    void repeatedComparisonUsesSharedFoldsAndExactProbabilities() throws Exception {
        assertEquals(0.5, ModelComparisonService.twoSidedTProbability(1.0, 1), 1e-12);
        assertEquals(1.0 - 2.0 / Math.sqrt(6.0), ModelComparisonService.twoSidedTProbability(2.0, 2), 1e-12);
        assertEquals(0.05, ModelComparisonService.twoSidedTProbability(2.228138851986273, 10), 1e-9);
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            String baseline = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.rules.ZeroR")))
                    .get("model_id").asText();
            String tree = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            Map<String, Object> arguments = Map.of(
                    "dataset_id", datasetId, "model_ids", List.of(baseline, tree),
                    "folds", 3, "runs", 2, "seed", 42, "confidence", 0.95);
            JsonNode result = success(fixture.tools.compareModels(arguments));
            assertEquals("repeated_stratified_cross_validation", result.get("protocol").asText());
            assertTrue(result.get("shared_folds").booleanValue());
            assertTrue(result.get("fold_safe_preprocessing").booleanValue());
            assertEquals(6, result.get("fold_comparisons").intValue());
            assertFalse(result.get("partial").booleanValue());
            assertEquals(6, result.get("models").get(0).get("fold_scores").size());
            assertEquals(6, result.get("models").get(1).get("fold_scores").size());
            JsonNode comparison = result.get("comparisons").get(0);
            assertEquals(5, comparison.get("degrees_of_freedom").intValue());
            assertTrue(comparison.get("p_value").doubleValue() >= 0.0);
            assertTrue(comparison.get("p_value").doubleValue() <= 1.0);
            assertNotNull(comparison.get("confidence_interval_low"));
            assertEquals(result, success(fixture.tools.compareModels(arguments)));

            String regressionId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff")))
                    .get("dataset_id").asText();
            String zero = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", regressionId, "classifier_class", "weka.classifiers.rules.ZeroR")))
                    .get("model_id").asText();
            String linear = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", regressionId, "classifier_class", "weka.classifiers.functions.LinearRegression")))
                    .get("model_id").asText();
            JsonNode regression = success(fixture.tools.compareModels(Map.of(
                    "dataset_id", regressionId, "model_ids", List.of(zero, linear),
                    "metric", "rmse", "folds", 2, "runs", 1)));
            assertEquals(2, regression.get("fold_comparisons").intValue());
            assertEquals("rmse", regression.get("metric").asText());
        }
    }

    @Test
    void repeatedEvaluationReturnsEachRunAndAggregateMetrics() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            JsonNode repeated = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48",
                    "folds", 3, "repeats", 2, "seed", 42)));
            assertEquals(2, repeated.get("repeats").intValue());
            assertEquals(2, repeated.get("repeat_metrics").size());
            assertEquals("mean_of_repeats", repeated.get("aggregation").asText());
            double expectedAccuracy = (repeated.get("repeat_metrics").get(0).get("accuracy_percent").doubleValue()
                    + repeated.get("repeat_metrics").get(1).get("accuracy_percent").doubleValue()) / 2;
            assertEquals(expectedAccuracy, repeated.get("accuracy_percent").doubleValue(), 1e-12);
            double total = 0;
            for (JsonNode row : repeated.get("confusion_matrix")) {
                for (JsonNode value : row) {
                    total += value.doubleValue();
                }
            }
            assertEquals(60, total, 0.0);
            for (JsonNode perClass : repeated.get("per_class")) {
                assertEquals(20, perClass.get("support").doubleValue(), 0.0);
            }
            JsonNode holdout = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48",
                    "mode", "holdout", "repeats", 2, "seed", 42)));
            assertEquals(2, holdout.get("repeat_metrics").size());
            assertNotNull(holdout.get("accuracy_percent"));
        }
    }

    @Test
    void thresholdAndMarginAnalysisComputesExactCosts() throws Exception {
        try (Fixture fixture = new Fixture()) {
            Files.writeString(directory.resolve("binary.arff"), """
                    @relation binary
                    @attribute x numeric
                    @attribute class {a,b}
                    @data
                    0,a
                    0.1,a
                    0.2,a
                    0.3,a
                    1,b
                    1.1,b
                    """);
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "binary.arff")))
                    .get("dataset_id").asText();
            String majority = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.rules.ZeroR")))
                    .get("model_id").asText();
            JsonNode analysis = success(fixture.tools.thresholdAnalysis(Map.of(
                    "model_id", majority, "dataset_id", datasetId, "positive_class", "b",
                    "thresholds", "0.2,0.5", "cost_fp", 2, "cost_fn", 3, "bins", 10)));
            assertEquals(0.5, analysis.get("best_threshold").doubleValue(), 0.0);
            assertEquals(6, analysis.get("minimum_cost").doubleValue(), 0.0);
            JsonNode best = analysis.get("best");
            assertEquals(0, best.get("tp").intValue());
            assertEquals(0, best.get("fp").intValue());
            assertEquals(2, best.get("fn").intValue());
            assertEquals(4, best.get("tn").intValue());
            assertEquals(6, analysis.get("scored_instances").intValue());
            int margins = 0;
            for (JsonNode bin : analysis.get("margin_histogram")) {
                margins += bin.get("correct").intValue() + bin.get("incorrect").intValue();
            }
            assertEquals(6, margins);

            String naive = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.bayes.NaiveBayes")))
                    .get("model_id").asText();
            JsonNode curve = success(fixture.tools.thresholdAnalysis(Map.of(
                    "model_id", naive, "dataset_id", datasetId, "positive_class", "b")));
            assertEquals(21, curve.get("thresholds").size());
            assertEquals(0, curve.get("minimum_cost").doubleValue(), 0.0);
            assertEquals(6, curve.get("scored_instances").intValue());
            assertTrue(fixture.tools.thresholdAnalysis(Map.of(
                    "model_id", naive, "dataset_id", datasetId, "thresholds", "NaN")).isError());
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
            for (String name : List.of("iris.arff", "regression.arff")) {
                try (InputStream input = ModelComparisonServiceTest.class.getResourceAsStream("/" + name)) {
                    assertNotNull(input);
                    Files.copy(input, directory.resolve(name), StandardCopyOption.REPLACE_EXISTING);
                }
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
