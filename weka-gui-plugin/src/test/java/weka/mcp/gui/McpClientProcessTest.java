package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import java.nio.file.Files;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class McpClientProcessTest {
    @Test
    void completeWorkflowUsesRealMcpProcess() throws Exception {
        PluginWorkspace workspace = TestSupport.workspace();
        TestSupport.iris();
        try (McpClientProcess client = workspace.newClient()) {
            client.start();
            assertTrue(client.isRunning());
            String datasetId = client.callTool("weka_load_dataset", Map.of(
                    "path", "iris.arff", "class_index", 5)).get("dataset_id").asText();
            assertEquals(30, client.callTool("weka_profile_dataset", Map.of("dataset_id", datasetId))
                    .get("instances").intValue());
            JsonNode trained = client.callTool("weka_train_classifier", Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48", "persist", true));
            String modelId = trained.get("model_id").asText();
            String experimentId = trained.get("experiment_id").asText();
            assertNotNull(experimentId);
            assertNotNull(client.callTool("weka_evaluate_model", Map.of(
                    "model_id", modelId, "test_dataset_id", datasetId)).get("accuracy_percent"));

            String instance = "{\"sepallength\":5.1,\"sepalwidth\":3.5,\"petallength\":1.4,\"petalwidth\":0.2}";
            JsonNode expected = client.callTool("weka_predict", Map.of(
                    "model_id", modelId, "instance_json", instance)).get("predictions");
            assertEquals("Iris-setosa", expected.get(0).get("predicted_value").asText());
            assertTrue(client.callTool("weka_export_model", Map.of(
                    "model_id", modelId, "path", "exports/j48.model")).get("gui_compatible").booleanValue());
            JsonNode imported = client.callTool("weka_import_model", Map.of(
                    "path", "exports/j48.model", "dataset_id", datasetId, "trusted", true));
            assertFalse(imported.get("retrainable").booleanValue());
            assertEquals(expected, client.callTool("weka_predict", Map.of(
                    "model_id", imported.get("model_id").asText(), "instance_json", instance))
                    .get("predictions"));
            assertEquals(expected, client.callTool("weka_predict", Map.of(
                    "model_id", client.callTool("weka_load_experiment_model", Map.of(
                            "experiment_id", experimentId)).get("model_id").asText(),
                    "instance_json", instance)).get("predictions"));

            String zeroR = client.callTool("weka_train_classifier", Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.rules.ZeroR"))
                    .get("model_id").asText();
            JsonNode comparison = client.callTool("weka_compare_models", Map.of(
                    "dataset_id", datasetId, "model_ids", new String[]{modelId, zeroR},
                    "folds", 2, "runs", 1));
            assertTrue(comparison.get("shared_folds").booleanValue());
            assertTrue(comparison.get("comparisons").get(0).get("p_value").doubleValue() >= 0);
            assertTrue(comparison.get("comparisons").get(0).get("p_value").doubleValue() <= 1);

            String clustererId = client.callTool("weka_cluster", Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3)).get("clusterer_id").asText();
            assertEquals(3, client.callTool("weka_cluster_evaluate", Map.of(
                    "clusterer_id", clustererId, "dataset_id", datasetId)).get("num_clusters").intValue());
            String logistic = client.callTool("weka_train_classifier", Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.functions.Logistic"))
                    .get("model_id").asText();
            client.callTool("weka_export_model", Map.of(
                    "model_id", logistic, "path", "exports/logistic.pmml", "format", "pmml"));
            assertTrue(Files.readString(workspace.workspaceDirectory().resolve("data/exports/logistic.pmml"))
                    .contains("<PMML"));
            assertThrows(McpClientProcess.McpException.class,
                    () -> client.callTool("weka_describe_dataset", Map.of("dataset_id", "invalid")));
            client.close();
            assertFalse(client.isRunning());
            assertEquals(0, client.exitCode());
            assertTrue(client.stderr().contains("started with 43 tools"));
        }
    }

    @Test
    void incrementalUpdateCanBeRestored() throws Exception {
        PluginWorkspace workspace = TestSupport.workspace();
        TestSupport.iris();
        TestSupport.writeUpdate();
        try (McpClientProcess client = workspace.newClient()) {
            client.start();
            JsonNode stream = client.callTool("weka_train_stream", Map.of(
                    "path", "iris.arff", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable"));
            JsonNode updated = client.callTool("weka_update_model", Map.of(
                    "model_id", stream.get("model_id").asText(), "path", "update.arff"));
            assertEquals(2, updated.get("updated_instances").intValue());
            JsonNode restored = client.callTool("weka_load_experiment_model", Map.of(
                    "experiment_id", updated.get("experiment_id").asText()));
            String instance = "{\"sepallength\":5.1,\"sepalwidth\":3.5,\"petallength\":1.4,\"petalwidth\":0.2}";
            assertEquals(client.callTool("weka_predict", Map.of(
                            "model_id", stream.get("model_id").asText(), "instance_json", instance))
                    .get("predictions"), client.callTool("weka_predict", Map.of(
                            "model_id", restored.get("model_id").asText(), "instance_json", instance))
                    .get("predictions"));
            client.close();
            assertEquals(0, client.exitCode());
        }
    }
}
