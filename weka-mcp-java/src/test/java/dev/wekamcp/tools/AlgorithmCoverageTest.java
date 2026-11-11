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
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AlgorithmCoverageTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void initializeWeka() {
        WekaPackageManager.loadPackages(false, true, false);
    }

    @Test
    void expandedClassifiersTrainOnAppropriateTasks() throws Exception {
        try (Fixture fixture = new Fixture()) {
            JsonNode algorithms = success(fixture.tools.listAlgorithms(Map.of("kind", "classifier")));
            assertEquals(36, algorithms.get("count").intValue());
            boolean foundIncremental = false;
            for (JsonNode algorithm : algorithms.get("classifiers")) {
                assertFalse(algorithm.get("capabilities").isEmpty());
                if (algorithm.get("class").asText().equals("weka.classifiers.trees.HoeffdingTree")) {
                    foundIncremental = algorithm.get("incremental").booleanValue();
                }
            }
            assertTrue(foundIncremental);
            assertEquals(62, success(fixture.tools.listAlgorithms(Map.of())).get("count").intValue());
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff")))
                    .get("dataset_id").asText();
            for (String classifier : List.of(
                    "weka.classifiers.rules.OneR", "weka.classifiers.trees.DecisionStump",
                    "weka.classifiers.rules.JRip", "weka.classifiers.rules.PART",
                    "weka.classifiers.rules.DecisionTable", "weka.classifiers.bayes.BayesNet",
                    "weka.classifiers.bayes.NaiveBayesMultinomial",
                    "weka.classifiers.bayes.NaiveBayesMultinomialUpdateable",
                    "weka.classifiers.trees.RandomTree", "weka.classifiers.trees.REPTree",
                    "weka.classifiers.trees.HoeffdingTree", "weka.classifiers.functions.SimpleLogistic",
                    "weka.classifiers.functions.MultilayerPerceptron", "weka.classifiers.lazy.KStar",
                    "weka.classifiers.lazy.LWL", "weka.classifiers.meta.Bagging",
                    "weka.classifiers.meta.LogitBoost", "weka.classifiers.meta.RandomCommittee",
                    "weka.classifiers.meta.RandomSubSpace")) {
                JsonNode trained = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", datasetId, "classifier_class", classifier)));
                assertEquals(classifier, trained.get("classifier_class").asText());
                assertNotNull(trained.get("model_id"));
            }
            String regressionId = success(fixture.tools.loadDataset(Map.of("path", "regression.arff")))
                    .get("dataset_id").asText();
            for (String classifier : List.of(
                    "weka.classifiers.rules.M5Rules", "weka.classifiers.trees.M5P",
                    "weka.classifiers.meta.AdditiveRegression", "weka.classifiers.meta.RegressionByDiscretization")) {
                JsonNode trained = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", regressionId, "classifier_class", classifier)));
                assertEquals(classifier, trained.get("classifier_class").asText());
                JsonNode evaluated = success(fixture.tools.evaluate(Map.of(
                        "dataset_id", regressionId, "model_id", trained.get("model_id").asText(), "folds", 2)));
                assertNotNull(evaluated.get("rmse"));
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
                    1,1,b
                    1.1,1.2,b
                    1.2,1.1,b
                    1.3,1.2,b
                    """);
            String binaryId = success(fixture.tools.loadDataset(Map.of("path", "binary.arff")))
                    .get("dataset_id").asText();
            assertNotNull(success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", binaryId, "classifier_class", "weka.classifiers.functions.VotedPerceptron")))
                    .get("model_id"));
        }
    }

    @Test
    void textAssociationsAndClustersCoverTheLocalWorkflow() throws Exception {
        try (Fixture fixture = new Fixture()) {
            Files.writeString(directory.resolve("text.arff"), """
                    @relation text
                    @attribute content string
                    @attribute class {positive,negative}
                    @data
                    'good nice excellent',positive
                    'good pleasant',positive
                    'nice excellent good',positive
                    'pleasant good',positive
                    'bad awful terrible',negative
                    'bad poor',negative
                    'awful terrible bad',negative
                    'poor bad',negative
                    """);
            String textId = success(fixture.tools.loadDataset(Map.of("path", "text.arff")))
                    .get("dataset_id").asText();
            JsonNode textModel = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", textId, "classifier_class", "weka.classifiers.bayes.NaiveBayesMultinomial",
                    "auto_preprocess", true)));
            assertEquals("weka.filters.unsupervised.attribute.StringToWordVector",
                    textModel.get("preprocessing").get(0).get("filterClass").asText());
            String instance = ToolSupport.JSON.writeValueAsString(Map.of("content", "good excellent"));
            assertNotNull(success(fixture.tools.predict(Map.of(
                    "model_id", textModel.get("model_id").asText(), "instance_json", instance)))
                    .get("predictions").get(0).get("predicted_value"));

            String weatherId = success(fixture.tools.loadDataset(Map.of("path", "weather.nominal.arff")))
                    .get("dataset_id").asText();
            for (String algorithm : List.of("apriori", "fpgrowth")) {
                JsonNode rules = success(fixture.tools.associate(Map.of(
                        "dataset_id", weatherId, "algorithm", algorithm,
                        "min_support", 0.1, "min_confidence", 0.1, "max_rules", 5)));
                assertEquals(algorithm, rules.get("algorithm").asText());
                assertTrue(rules.get("rule_count").intValue() > 0);
                assertTrue(rules.get("rule_count").intValue() <= 5);
                assertNotNull(rules.get("rules").get(0).get("lift"));
            }
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            for (String clusterer : List.of("weka.clusterers.FarthestFirst", "weka.clusterers.Canopy",
                    "weka.clusterers.Cobweb")) {
                JsonNode trained = success(fixture.tools.cluster(clusterer.endsWith("FarthestFirst")
                        ? Map.of("dataset_id", datasetId, "clusterer_class", clusterer, "options", List.of("-N", "3"))
                        : Map.of("dataset_id", datasetId, "clusterer_class", clusterer)));
                assertTrue(trained.get("num_clusters").intValue() > 0);
            }
            String clustererId = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3))).get("clusterer_id").asText();
            JsonNode evaluation = success(fixture.tools.clusterEvaluate(Map.of(
                    "clusterer_id", clustererId, "dataset_id", datasetId, "max_samples", 12)));
            assertEquals(30, evaluation.get("instances").intValue());
            assertEquals(3, evaluation.get("class_to_cluster").size());
            assertEquals(3, evaluation.get("silhouette").get("per_cluster").size());
            assertTrue(evaluation.get("silhouette").get("samples").intValue() > 0);
            assertTrue(evaluation.get("silhouette").get("samples").intValue() <= 12);
            assertTrue(evaluation.get("silhouette").get("sampled").booleanValue());
            assertTrue(evaluation.get("silhouette").get("mean").doubleValue() >= -1.0);
            assertTrue(evaluation.get("silhouette").get("mean").doubleValue() <= 1.0);
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
            for (String name : List.of("iris.arff", "regression.arff", "weather.nominal.arff")) {
                try (InputStream input = AlgorithmCoverageTest.class.getResourceAsStream("/" + name)) {
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
