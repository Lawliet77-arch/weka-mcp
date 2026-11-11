package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.explorer.Explorer;

import javax.swing.SwingUtilities;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExplorerClusterOperationsTest {
    private static final JsonMapper JSON = new JsonMapper();

    @Test
    @Timeout(60)
    void simpleKMeansRunsInsideExplorerClusterPanel() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        try {
            ObjectNode arguments = JSON.createObjectNode();
            arguments.put("clusterer_class", "weka.clusterers.SimpleKMeans");
            arguments.put("num_clusters", 3);
            arguments.put("seed", 1);
            assertEquals("RUNNING", session.call(ExplorerSession.CLUSTER, arguments).get("status").asText());
            JsonNode result = session.call(ExplorerSession.CLUSTER_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(result.get("complete").booleanValue());
            assertTrue(result.get("output").asText().contains("Clustered Instances"));
            assertFalse(result.get("clusterer_running").booleanValue());
            assertEquals("COMPLETE", result.get("result_status").asText());
            assertFalse(result.get("dataset_changed_since_training").booleanValue());
            assertFalse(result.get("operation_id").asText().isBlank());
            assertTrue(result.get("evaluation_metrics").get("within_cluster_squared_error").doubleValue() > 0);
            JsonNode state = session.call(ExplorerSession.STATE, null);
            assertEquals("Cluster", state.get("active_tab").asText());
            assertEquals(1, state.get("cluster_result_count").intValue());
            assertFalse(state.get("task_running").booleanValue());
            assertFalse(session.call(ExplorerSession.CLUSTER_STOP, null).get("stop_requested").booleanValue());
            assertEquals(21, ExplorerSession.toolSpecifications().size());
        } finally {
            session.close();
        }
    }

    @Test
    @Timeout(90)
    void additionalNativeClusterersProduceResults() throws Exception {
        for (String className : new String[]{"weka.clusterers.FarthestFirst",
                "weka.clusterers.Canopy", "weka.clusterers.Cobweb"}) {
            Explorer explorer = new Explorer();
            explorer.getPreprocessPanel().setInstances(TestSupport.iris());
            SwingUtilities.invokeAndWait(() -> { });
            ExplorerSession session = new ExplorerSession(explorer);
            try {
                ObjectNode arguments = JSON.createObjectNode();
                arguments.put("clusterer_class", className);
                if (!className.endsWith("Cobweb")) {
                    arguments.put("seed", 1);
                }
                session.call(ExplorerSession.CLUSTER, arguments);
                JsonNode result = session.call(ExplorerSession.CLUSTER_RESULTS,
                        JSON.createObjectNode().put("wait_seconds", 30));
                assertTrue(result.get("complete").booleanValue(), className);
                assertFalse(result.get("output").asText().isBlank(), className);
                assertFalse(result.get("clusterer_running").booleanValue(), className);
            } finally {
                session.close();
            }
        }
    }

    @Test
    @Timeout(90)
    void clusteringCanBeCancelledAndBlocksDatasetChanges() throws Exception {
        Explorer explorer = new Explorer();
        Instances base = TestSupport.iris();
        Instances data = new Instances(base, 0);
        Random random = new Random(1);
        for (int index = 0; index < 100_000; index++) {
            Instance instance = (Instance) base.instance(index % base.numInstances()).copy();
            instance.setValue(0, random.nextDouble() * 10);
            instance.setValue(1, random.nextDouble() * 10);
            data.add(instance);
        }
        explorer.getPreprocessPanel().setInstances(data);
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        try {
            ObjectNode arguments = JSON.createObjectNode();
            arguments.put("clusterer_class", "weka.clusterers.SimpleKMeans");
            arguments.put("num_clusters", 50);
            arguments.putArray("options").add("-I").add("1000000");
            session.call(ExplorerSession.CLUSTER, arguments);
            assertTrue(session.call(ExplorerSession.STATE, null)
                    .get("clusterer_running").booleanValue());
            assertThrows(IllegalStateException.class, () -> session.call(ExplorerSession.SET_CLASS,
                    JSON.createObjectNode().put("class_index", 1)));
            assertThrows(IllegalStateException.class, () -> session.call(ExplorerSession.UNDO, null));
            assertTrue(session.call(ExplorerSession.CLUSTER_STOP, null)
                    .get("stop_requested").booleanValue());
            JsonNode cancelled = session.call(ExplorerSession.CLUSTER_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertFalse(cancelled.get("clusterer_running").booleanValue());
            assertEquals("CANCELLED", cancelled.get("result_status").asText());
            assertFalse(cancelled.get("complete").booleanValue());
        } finally {
            session.close();
        }
    }
}
