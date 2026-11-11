package dev.wekamcp.tools;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WekaToolsTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void initializeWeka() {
        WekaPackageManager.loadPackages(false, true, false);
    }

    @Test
    void completeClassificationAndClusteringWorkflow() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            JsonNode loaded = success(fixture.tools.loadDataset(Map.of(
                    "path", "iris.arff", "format", "arff", "class_index", 5)));
            String datasetId = loaded.get("dataset_id").asText();
            assertEquals(30, loaded.get("instances").intValue());
            assertEquals(5, loaded.get("attributes").intValue());

            JsonNode description = success(fixture.tools.describeDataset(Map.of("dataset_id", datasetId)));
            assertEquals(5, description.get("attributes").intValue());
            assertEquals(5, description.get("attribute_details").size());
            assertEquals(3, description.get("attribute_details").get(4).get("nominal_values").size());
            assertEquals(18, success(fixture.tools.listFilters(Map.of())).get("filters").size());

            JsonNode filtered = success(fixture.tools.applyFilter(Map.of(
                    "dataset_id", datasetId,
                    "filter_class", "weka.filters.unsupervised.attribute.Remove",
                    "options", List.of("-R", "1"))));
            assertEquals(4, filtered.get("attributes").intValue());

            JsonNode trained = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48",
                    "options", List.of("-C", "0.25"))));
            String modelId = trained.get("model_id").asText();
            assertEquals("class", trained.get("class_attribute").asText());

            JsonNode evaluation = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId, "model_id", modelId, "folds", 3)));
            assertEquals(3, evaluation.get("folds").intValue());
            assertEquals(3, evaluation.get("confusion_matrix").size());
            assertNotNull(evaluation.get("accuracy_percent"));

            JsonNode holdout = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48",
                    "mode", "holdout",
                    "train_percent", 70)));
            assertEquals(9, holdout.get("test_instances").intValue());

            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode predicted = success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", instance)));
            assertEquals("Iris-setosa", predicted.get("predictions").get(0).get("predicted_value").asText());
            assertEquals(3, predicted.get("predictions").get(0).get("class_distribution").size());

            JsonNode batch = success(fixture.tools.predict(Map.of("model_id", modelId, "dataset_id", datasetId)));
            assertEquals(30, batch.get("count").intValue());

            JsonNode clustered = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId,
                    "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3,
                    "options", List.of("-S", "42"))));
            assertEquals(3, clustered.get("num_clusters").intValue());
            assertEquals(30, clustered.get("assignments").size());

            assertTrue(fixture.tools.release(Map.of("resource_id", modelId)).isError() == false);
            assertTrue(fixture.tools.predict(Map.of("model_id", modelId, "instance_json", instance)).isError());
        }
    }

    @Test
    void returnsStructuredErrorsForInvalidInputs() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            assertTrue(fixture.tools.loadDataset(Map.of("path", "../iris.arff")).isError());
            Path outsideFile = Files.createTempFile("weka-outside", ".arff");
            try {
                CallToolResult outsideResult = fixture.tools.loadDataset(Map.of("path", outsideFile.toString()));
                assertTrue(outsideResult.isError());
                assertTrue(text(outsideResult).contains("data root"), () -> text(outsideResult));
            } finally {
                Files.deleteIfExists(outsideFile);
            }
            assertTrue(fixture.tools.describeDataset(Map.of("dataset_id", "not-a-uuid")).isError());
            assertTrue(fixture.tools.describeDataset(Map.of("dataset_id", UUID.randomUUID().toString())).isError());

            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            assertTrue(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.functions.DL4J")).isError());
            assertTrue(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48",
                    "options", List.of("-X", "1"))).isError());
            assertTrue(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48",
                    "options", Collections.nCopies(21, "-U"))).isError());

            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            String invalidInstance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2,
                    "class", "unknown"));
            CallToolResult result = fixture.tools.predict(Map.of("model_id", modelId, "instance_json", invalidInstance));
            assertTrue(result.isError());
            assertTrue(text(result).contains("Invalid nominal value"));
        }
    }

    @Test
    void enforcesDatasetRegistryAndOutputLimits() throws Exception {
        try (Fixture fixture = new Fixture(directory, "maxInstances=10\n")) {
            CallToolResult result = fixture.tools.loadDataset(Map.of("path", "iris.arff"));
            assertTrue(result.isError());
            assertTrue(text(result).contains("maxInstances"));
        }
        try (Fixture fixture = new Fixture(directory, "maxDatasets=1\n")) {
            assertFalse(fixture.tools.loadDataset(Map.of("path", "iris.arff")).isError());
            CallToolResult result = fixture.tools.loadDataset(Map.of("path", "iris.arff"));
            assertTrue(result.isError());
            assertTrue(text(result).contains("weka_release"));
        }
        try (Fixture fixture = new Fixture(directory, "maxOutputBytes=100\n")) {
            CallToolResult result = fixture.tools.loadDataset(Map.of("path", "iris.arff"));
            assertTrue(result.isError());
            assertTrue(text(result).contains("maxOutputBytes"));
        }
    }

    @Test
    void evaluatesNumericClassWithRegressionMetrics() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff")))
                    .get("dataset_id").asText();
            JsonNode evaluation = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.rules.ZeroR",
                    "folds", 2)));
            assertEquals("numeric", evaluation.get("class_type").asText());
            assertNotNull(evaluation.get("mae"));
            assertNotNull(evaluation.get("rmse"));
            assertFalse(evaluation.has("confusion_matrix"));
        }
    }

    @Test
    void outputLimitFailureDoesNotConsumeDatasetCapacity() throws Exception {
        try (Fixture fixture = new Fixture(directory, "maxDatasets=1\nmaxOutputBytes=100\n")) {
            CallToolResult first = fixture.tools.loadDataset(Map.of("path", "iris.arff"));
            assertTrue(first.isError());
            assertTrue(text(first).contains("maxOutputBytes"));

            CallToolResult second = fixture.tools.loadDataset(Map.of("path", "iris.arff"));
            assertTrue(second.isError());
            assertTrue(text(second).contains("maxOutputBytes"), () -> text(second));
        }
    }

    @Test
    void holdoutDoesNotDependOnUnusedDefaultFoldCount() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            Files.writeString(directory.resolve("small.arff"), """
                    @relation small
                    @attribute x numeric
                    @attribute class {a,b}
                    @data
                    1,a
                    2,a
                    3,a
                    4,b
                    5,b
                    6,b
                    """);
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "small.arff")))
                    .get("dataset_id").asText();
            JsonNode evaluation = success(fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.rules.ZeroR",
                    "mode", "holdout",
                    "train_percent", 67)));
            assertEquals(2, evaluation.get("test_instances").intValue());
            assertTrue(evaluation.get("stratified").booleanValue());

            JsonNode automatic = success(fixture.tools.autoClassify(Map.of("dataset_id", datasetId)));
            assertEquals(2, automatic.get("validation").get("folds").intValue());
            assertEquals(6, automatic.get("candidates").size());
        }
    }

    @Test
    void modelEvaluationRejectsIgnoredReplacementOptions() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            CallToolResult result = fixture.tools.evaluate(Map.of(
                    "dataset_id", datasetId,
                    "model_id", modelId,
                    "options", List.of("-U")));
            assertTrue(result.isError(), () -> text(result));
            assertTrue(text(result).contains("options"));
        }
    }

    @Test
    void predictionRejectsTrailingJsonAndNonFiniteNumbers() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));

            CallToolResult trailing = fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", instance + " trailing"));
            assertTrue(trailing.isError(), () -> text(trailing));

            CallToolResult nonFinite = fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", instance.replace("5.1", "\"NaN\"")));
            assertTrue(nonFinite.isError(), () -> text(nonFinite));
            assertTrue(text(nonFinite).contains("finite"));
        }
    }

    @Test
    void automaticClassificationCreatesReusablePipelines() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            JsonNode automatic = success(fixture.tools.autoClassify(Map.of("dataset_id", datasetId, "folds", 3)));
            assertTrue(automatic.get("auto_selected").booleanValue());
            assertEquals(6, automatic.get("candidates").size());
            assertFalse(automatic.get("classifier_class").asText().isBlank());
            assertEquals("accuracy", automatic.get("selection_metric").asText());
            assertNotNull(automatic.get("selection_score"));
            assertTrue(automatic.get("auto_preprocess").booleanValue());
            assertTrue(automatic.get("preprocessing").size() >= 1);
            assertEquals(3, automatic.get("evaluation").get("per_class").size());
            assertTrue(automatic.get("evaluation").get("per_class").get(0).has("precision"));
            assertTrue(automatic.get("evaluation").get("per_class").get(0).has("auc"));

            String modelId = automatic.get("model_id").asText();
            JsonNode independent = success(fixture.tools.evaluateModel(Map.of(
                    "model_id", modelId, "test_dataset_id", datasetId)));
            assertTrue(independent.get("independent_test").booleanValue());
            assertNotNull(independent.get("accuracy_percent"));

            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            assertEquals("Iris-setosa", success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", instance)))
                    .get("predictions").get(0).get("predicted_value").asText());

            JsonNode logistic = success(fixture.tools.autoClassify(Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.functions.Logistic",
                    "options", List.of("-M", "100"),
                    "folds", 3)));
            assertEquals(2, logistic.get("preprocessing").size());
            assertEquals("weka.filters.unsupervised.attribute.Standardize",
                    logistic.get("preprocessing").get(1).get("filterClass").asText());
            assertFalse(fixture.tools.predict(Map.of(
                    "model_id", logistic.get("model_id").asText(), "instance_json", instance)).isError());

            JsonNode resources = success(fixture.tools.listResources(Map.of()));
            assertEquals(1, resources.get("dataset_count").intValue());
            assertEquals(2, resources.get("model_count").intValue());
            assertEquals(0, resources.get("clusterer_count").intValue());
        }
    }

    @Test
    void expandedAlgorithmsSupportClassificationAndRegression() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            for (String classifier : List.of(
                    "weka.classifiers.functions.SMO",
                    "weka.classifiers.lazy.IBk",
                    "weka.classifiers.meta.AdaBoostM1")) {
                JsonNode model = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", datasetId,
                        "classifier_class", classifier,
                        "auto_preprocess", true)));
                assertEquals(classifier, model.get("classifier_class").asText());
                assertNotNull(model.get("model_id"));
            }

            String regressionId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff")))
                    .get("dataset_id").asText();
            for (String classifier : List.of(
                    "weka.classifiers.functions.LinearRegression",
                    "weka.classifiers.functions.SMOreg")) {
                JsonNode model = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", regressionId,
                        "classifier_class", classifier,
                        "auto_preprocess", true)));
                JsonNode evaluation = success(fixture.tools.evaluate(Map.of(
                        "dataset_id", regressionId, "model_id", model.get("model_id").asText(), "folds", 2)));
                assertNotNull(evaluation.get("rmse"));
                assertNotNull(evaluation.get("rae_percent"));
            }
        }
    }

    @Test
    void clustererReusesLabelRemovalForNewPredictions() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            JsonNode kMeans = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId,
                    "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3)));
            assertEquals(4, kMeans.get("attributes").intValue());
            assertEquals(5, kMeans.get("removed_class_index").intValue());
            String clustererId = kMeans.get("clusterer_id").asText();
            assertEquals(30, success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", clustererId, "dataset_id", datasetId))).get("count").intValue());

            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode assigned = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", clustererId, "instance_json", instance)));
            assertTrue(assigned.get("predictions").get(0).has("cluster"));

            JsonNode em = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId,
                    "clusterer_class", "weka.clusterers.EM",
                    "num_clusters", 3,
                    "options", List.of("-I", "20", "-S", "42"))));
            assertNotNull(em.get("log_likelihood"));
            JsonNode distribution = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", em.get("clusterer_id").asText(),
                    "instance_json", instance,
                    "include_distribution", true)));
            assertEquals(3, distribution.get("predictions").get(0).get("cluster_distribution").size());

            JsonNode resources = success(fixture.tools.listResources(Map.of()));
            assertEquals(2, resources.get("clusterer_count").intValue());
            assertEquals(30, resources.get("clusterers").get(0).get("instances").intValue());
        }
    }

    @Test
    void ordinaryModelsAndClusterersPersistAndRestore() throws Exception {
        try (Fixture fixture = new Fixture(directory, "")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            String instance = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode trained = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")));
            JsonNode saved = success(fixture.tools.saveModel(Map.of("model_id", trained.get("model_id").asText())));
            String experimentId = saved.get("experiment_id").asText();
            assertEquals("CLASSIFIER", success(fixture.tools.getExperiment(Map.of(
                    "experiment_id", experimentId))).get("kind").asText());
            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of("experiment_id", experimentId)));
            assertEquals(success(fixture.tools.predict(Map.of(
                            "model_id", trained.get("model_id").asText(), "instance_json", instance))).get("predictions"),
                    success(fixture.tools.predict(Map.of(
                            "model_id", restored.get("model_id").asText(), "instance_json", instance))).get("predictions"));

            JsonNode clustered = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3, "options", List.of("-S", "42"), "persist", true)));
            String clustererExperiment = clustered.get("experiment_id").asText();
            JsonNode expected = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", clustered.get("clusterer_id").asText(), "dataset_id", datasetId)));
            JsonNode loaded = success(fixture.tools.loadClusterer(Map.of("experiment_id", clustererExperiment)));
            JsonNode actual = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", loaded.get("clusterer_id").asText(), "dataset_id", datasetId)));
            assertEquals(expected.get("predictions"), actual.get("predictions"));
            assertEquals("CLUSTERER", success(fixture.tools.getExperiment(Map.of(
                    "experiment_id", clustererExperiment))).get("kind").asText());
            JsonNode savedAgain = success(fixture.tools.saveClusterer(Map.of(
                    "clusterer_id", loaded.get("clusterer_id").asText())));
            assertFalse(savedAgain.get("experiment_id").asText().isBlank());
        }
    }

    private static JsonNode success(CallToolResult result) throws Exception {
        assertFalse(result.isError(), () -> text(result));
        return ToolSupport.JSON.readTree(text(result));
    }

    private static String text(CallToolResult result) {
        return ((McpSchema.TextContent) result.content().get(0)).text();
    }

    private static final class Fixture implements AutoCloseable {
        private final TaskRunner tasks;
        private final WekaTools tools;

        private Fixture(Path directory, String additionalConfig) throws Exception {
            Files.createDirectories(directory);
            copyResource("iris.arff", directory.resolve("iris.arff"));
            copyResource("regression.arff", directory.resolve("regression.arff"));
            Path whitelistPath = Path.of("weka-whitelist.yaml").toAbsolutePath();
            Path searchSpacePath = Path.of("automl-search.yaml").toAbsolutePath();
            Path configPath = directory.resolve("config.properties");
            Files.writeString(configPath,
                    "dataRoot=" + directory.toString().replace('\\', '/') + "\n"
                            + "whitelistPath=" + whitelistPath.toString().replace('\\', '/') + "\n"
                            + "searchSpacePath=" + searchSpacePath.toString().replace('\\', '/') + "\n"
                            + additionalConfig);
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

        private static void copyResource(String name, Path target) throws Exception {
            try (InputStream input = WekaToolsTest.class.getResourceAsStream("/" + name)) {
                if (input == null) {
                    throw new IllegalStateException("Missing test resource " + name);
                }
                Files.copy(input, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }
}
