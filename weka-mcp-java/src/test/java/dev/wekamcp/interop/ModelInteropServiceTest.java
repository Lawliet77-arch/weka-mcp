package dev.wekamcp.interop;

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
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelInteropServiceTest {
    @TempDir
    Path directory;

    @Test
    void exportsAndImportsGuiClassifiersAndRawClusterers() throws Exception {
        try (Fixture fixture = new Fixture("")) {
            String datasetId = success(fixture.tools.loadDataset(Map.of("path", "iris.arff", "class_index", 5)))
                    .get("dataset_id").asText();
            String modelId = success(fixture.tools.trainClassifier(Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48")))
                    .get("model_id").asText();
            JsonNode exported = success(fixture.tools.exportModel(Map.of(
                    "model_id", modelId, "path", "exports/j48.model")));
            assertTrue(exported.get("gui_compatible").booleanValue());
            assertTrue(exported.get("header_included").booleanValue());
            assertTrue(exported.get("bytes").longValue() > 0);
            assertTrue(fixture.tools.exportModel(Map.of("model_id", modelId, "path", "exports/j48.model")).isError());
            assertTrue(fixture.tools.importModel(Map.of(
                    "path", "exports/j48.model", "dataset_id", datasetId)).isError());

            Map<String, Object> input = Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2);
            String instance = ToolSupport.JSON.writeValueAsString(input);
            JsonNode expected = success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", instance))).get("predictions");
            JsonNode imported = success(fixture.tools.importModel(Map.of(
                    "path", "exports/j48.model", "dataset_id", datasetId, "trusted", true)));
            assertFalse(imported.get("retrainable").booleanValue());
            String importedId = imported.get("model_id").asText();
            assertEquals(expected, success(fixture.tools.predict(Map.of(
                    "model_id", importedId, "instance_json", instance))).get("predictions"));
            assertTrue(fixture.tools.evaluate(Map.of("dataset_id", datasetId, "model_id", importedId)).isError());
            assertNotNull(success(fixture.tools.evaluateModel(Map.of(
                    "model_id", importedId, "test_dataset_id", datasetId))).get("accuracy_percent"));
            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of(
                    "experiment_id", imported.get("experiment_id").asText())));
            assertFalse(restored.get("retrainable").booleanValue());
            assertEquals(expected, success(fixture.tools.predict(Map.of(
                    "model_id", restored.get("model_id").asText(), "instance_json", instance))).get("predictions"));

            for (String classifier : List.of(
                    "weka.classifiers.trees.RandomForest", "weka.classifiers.functions.Logistic")) {
                String trained = success(fixture.tools.trainClassifier(Map.of(
                        "dataset_id", datasetId, "classifier_class", classifier))).get("model_id").asText();
                JsonNode expectedModel = success(fixture.tools.predict(Map.of(
                        "model_id", trained, "instance_json", instance))).get("predictions");
                success(fixture.tools.exportModel(Map.of(
                        "model_id", trained, "path", "exports/" + classifier.substring(classifier.lastIndexOf('.') + 1)
                                + ".model", "format", "raw_model")));
                JsonNode importedModel = success(fixture.tools.importModel(Map.of(
                        "path", "exports/" + classifier.substring(classifier.lastIndexOf('.') + 1) + ".model",
                        "dataset_id", datasetId, "trusted", true)));
                assertEquals(expectedModel, success(fixture.tools.predict(Map.of(
                        "model_id", importedModel.get("model_id").asText(), "instance_json", instance)))
                        .get("predictions"));
            }

            String clustererId = success(fixture.tools.cluster(Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3, "options", List.of("-S", "42")))).get("clusterer_id").asText();
            success(fixture.tools.exportModel(Map.of(
                    "clusterer_id", clustererId, "path", "exports/kmeans.model", "format", "raw_model")));
            JsonNode importedClusterer = success(fixture.tools.importModel(Map.of(
                    "path", "exports/kmeans.model", "dataset_id", datasetId, "trusted", true)));
            assertEquals(3, importedClusterer.get("num_clusters").intValue());
            JsonNode before = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", clustererId, "dataset_id", datasetId))).get("predictions");
            JsonNode after = success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", importedClusterer.get("clusterer_id").asText(),
                    "dataset_id", datasetId))).get("predictions");
            assertEquals(before, after);
            JsonNode loaded = success(fixture.tools.loadClusterer(Map.of(
                    "experiment_id", importedClusterer.get("experiment_id").asText())));
            assertEquals(after, success(fixture.tools.clusterPredict(Map.of(
                    "clusterer_id", loaded.get("clusterer_id").asText(),
                    "dataset_id", datasetId))).get("predictions"));
        }
    }

    @Test
    void updateFromFileMatchesDirectWekaAndRetainsLineage() throws Exception {
        try (Fixture fixture = new Fixture("")) {
            JsonNode trained = success(fixture.tools.trainStream(Map.of(
                    "path", "iris.arff", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable")));
            String modelId = trained.get("model_id").asText();
            writeUpdate(directory);
            JsonNode updated = success(fixture.tools.updateModel(Map.of("model_id", modelId, "path", "update.arff")));
            assertEquals(2, updated.get("updated_instances").intValue());
            String experimentId = updated.get("experiment_id").asText();
            JsonNode manifest = success(fixture.tools.getExperiment(Map.of("experiment_id", experimentId)));
            assertEquals(trained.get("experiment_id"), manifest.get("parentExperimentId"));
            assertEquals("UPDATE_SOURCE", manifest.get("datasetRole").asText());
            assertEquals("incremental_update", manifest.get("protocol").asText());

            Instances full = new DataSource(directory.resolve("iris.arff").toString()).getDataSet();
            full.setClassIndex(4);
            NaiveBayesUpdateable direct = new NaiveBayesUpdateable();
            direct.buildClassifier(new Instances(full, 0));
            for (int index = 0; index < full.numInstances(); index++) {
                if ((index + 1) % 5 != 0) {
                    direct.updateClassifier(full.instance(index));
                }
            }
            Instances updates = new DataSource(directory.resolve("update.arff").toString()).getDataSet();
            updates.setClassIndex(4);
            for (Instance instance : updates) {
                direct.updateClassifier(instance);
            }
            Instance instance = updates.instance(0);
            String json = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode prediction = success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", json))).get("predictions").get(0);
            assertEquals(full.classAttribute().value((int) direct.classifyInstance(instance)),
                    prediction.get("predicted_value").asText());
            double[] distribution = direct.distributionForInstance(instance);
            for (int index = 0; index < distribution.length; index++) {
                assertEquals(distribution[index], prediction.get("class_distribution")
                        .get(full.classAttribute().value(index)).doubleValue(), 0.0);
            }
            JsonNode restored = success(fixture.tools.loadExperimentModel(Map.of("experiment_id", experimentId)));
            assertEquals(prediction, success(fixture.tools.predict(Map.of(
                    "model_id", restored.get("model_id").asText(), "instance_json", json)))
                    .get("predictions").get(0));
        }
    }

    @Test
    void failedUpdateDoesNotMutateTheRegisteredModelAndRejectsBadInputs() throws Exception {
        try (Fixture fixture = new Fixture("maxExperiments=1\n")) {
            JsonNode trained = success(fixture.tools.trainStream(Map.of(
                    "path", "iris.arff", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable")));
            String modelId = trained.get("model_id").asText();
            String json = ToolSupport.JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode before = success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", json))).get("predictions");
            writeUpdate(directory);
            CallToolResult failed = fixture.tools.updateModel(Map.of("model_id", modelId, "path", "update.arff"));
            assertTrue(failed.isError(), () -> text(failed));
            assertTrue(text(failed).contains("Maximum number of experiments"));
            assertEquals(before, success(fixture.tools.predict(Map.of(
                    "model_id", modelId, "instance_json", json))).get("predictions"));

            Files.writeString(directory.resolve("invalid.model"), "not a serialized model");
            assertTrue(fixture.tools.importModel(Map.of(
                    "path", "invalid.model", "dataset_id", "00000000-0000-0000-0000-000000000000",
                    "trusted", true)).isError());
            assertTrue(fixture.tools.exportModel(Map.of("model_id", modelId, "path", "../escape.model")).isError());
        }
    }

    private static void writeUpdate(Path directory) throws Exception {
        Files.writeString(directory.resolve("update.arff"), """
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

        private Fixture(String extra) throws Exception {
            for (String name : List.of("iris.arff", "regression.arff")) {
                try (InputStream input = ModelInteropServiceTest.class.getResourceAsStream("/" + name)) {
                    assertNotNull(input);
                    Files.copy(input, directory.resolve(name), StandardCopyOption.REPLACE_EXISTING);
                }
            }
            Path configPath = directory.resolve("config.properties");
            Files.writeString(configPath,
                    "dataRoot=" + directory.toString().replace('\\', '/') + "\n"
                            + "whitelistPath=" + Path.of("weka-whitelist.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n"
                            + "searchSpacePath=" + Path.of("automl-search.yaml").toAbsolutePath().toString().replace('\\', '/') + "\n"
                            + extra);
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
