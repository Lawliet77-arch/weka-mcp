package dev.wekamcp.tools;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
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

class FeatureCompletionTest {
    @TempDir
    Path directory;

    @Test
    void pmmlAndCapabilityRecommendationsWork() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.functions.Logistic")))
                    .get("model_id").asText();
            JsonNode exported = success(fixture.tools.exportModel(Map.of(
                    "model_id", modelId, "path", "exports/logistic.pmml", "format", "pmml")));
            assertEquals("pmml", exported.get("format").asText());
            assertTrue(exported.get("header_included").booleanValue());
            assertFalse(exported.get("gui_compatible").booleanValue());
            assertFalse(exported.get("trusted_java_serialization").booleanValue());
            String xml = Files.readString(directory.resolve("exports/logistic.pmml"));
            assertTrue(xml.contains("<PMML"));
            org.w3c.dom.Document document = javax.xml.parsers.DocumentBuilderFactory.newInstance()
                    .newDocumentBuilder().parse(directory.resolve("exports/logistic.pmml").toFile());
            assertEquals("PMML", document.getDocumentElement().getTagName());
            assertTrue(document.getElementsByTagName("*").getLength() > 1);

            String tree = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            assertTrue(fixture.tools.exportModel(Map.of(
                    "model_id", tree, "path", "exports/tree.pmml", "format", "pmml")).isError());

            JsonNode recommendations = success(fixture.tools.recommend(Map.of(
                    "dataset_id", datasetId, "streaming", true, "limit", 20)));
            assertTrue(recommendations.get("requires_validation").booleanValue());
            assertEquals("weka_train_stream", recommendations.get("recommended_next_tool").asText());
            assertTrue(recommendations.get("recommendations").size() > 0);
            for (JsonNode recommendation : recommendations.get("recommendations")) {
                assertTrue(recommendation.get("incremental").booleanValue());
            }
            JsonNode ordinary = success(fixture.tools.recommend(Map.of("dataset_id", datasetId)));
            assertEquals("weka_auto_classify", ordinary.get("recommended_next_tool").asText());
            assertTrue(ordinary.get("recommendations").size() > 0);
        }
    }

    @Test
    void customCandidatesSupportNestedHalvingAndRandomSearch() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            List<Map<String, Object>> candidates = List.of(
                    Map.of("classifier_class", "weka.classifiers.trees.RandomTree", "options", List.of("-K", "2")),
                    Map.of("classifier_class", "weka.classifiers.bayes.NaiveBayes"));
            Map<String, Object> arguments = Map.of(
                    "dataset_id", datasetId, "candidates", candidates, "search", "halving", "protocol", "nested_cv",
                    "outer_folds", 2, "inner_folds", 2, "seed", 42L, "budgetSeconds", 120);
            JsonNode nested = success(fixture.tools.autoClassify(arguments));
            assertEquals(3, nested.get("candidate_count").intValue());
            assertEquals("weka.classifiers.rules.ZeroR",
                    nested.get("candidates").get(0).get("classifier_class").asText());
            assertEquals(3, nested.get("halving_stages").get(0).get("completed_candidates").intValue());
            assertEquals(2, nested.get("halving_stages").get(1).get("completed_candidates").intValue());
            assertEquals(1, nested.get("halving_stages").get(2).get("completed_candidates").intValue());
            assertTrue(nested.get("nested_cv_complete").booleanValue());
            assertEquals("full_data", nested.get("selection_source").asText());
            assertNotNull(nested.get("test_score"));

            Map<String, Object> randomArguments = Map.of(
                    "dataset_id", datasetId, "candidates", candidates, "search", "random",
                    "protocol", "holdout", "folds", 2, "seed", 42L);
            JsonNode first = success(fixture.tools.autoClassify(randomArguments));
            JsonNode second = success(fixture.tools.autoClassify(randomArguments));
            assertEquals(first.get("candidates"), second.get("candidates"));
            assertEquals(first.get("classifier_class"), second.get("classifier_class"));
        }
    }

    @Test
    void attributeSelectorsAndScatterMatrixCoverAnalysis() throws Exception {
        try (Fixture fixture = new Fixture()) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            for (String method : List.of("gain_ratio", "symmetrical_uncert", "one_r", "relief_f")) {
                JsonNode selection = success(fixture.tools.selectAttributes(Map.of(
                        "dataset_id", datasetId, "method", method, "max_attributes", 2)));
                assertEquals(method, selection.get("method").asText());
                assertEquals(2, selection.get("ranking").size());
                assertEquals(2, selection.get("selected_attributes").size());
                assertNotNull(selection.get("ranking").get(0).get("score"));
            }
            JsonNode matrix = success(fixture.tools.scatterMatrix(Map.of(
                    "dataset_id", datasetId, "max_points", 5)));
            assertEquals(6, matrix.get("plot_count").intValue());
            assertEquals(5, matrix.get("max_points").intValue());
            for (JsonNode plot : matrix.get("plots")) {
                assertEquals(5, plot.get("count").intValue());
            }
            assertTrue(fixture.tools.scatterMatrix(Map.of(
                    "dataset_id", datasetId, "attributes", List.of(1, 1))).isError());
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
            try (InputStream input = FeatureCompletionTest.class.getResourceAsStream("/iris.arff")) {
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
