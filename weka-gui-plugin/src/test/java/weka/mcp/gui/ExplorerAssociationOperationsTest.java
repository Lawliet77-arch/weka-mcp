package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.gui.explorer.Explorer;

import javax.swing.SwingUtilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExplorerAssociationOperationsTest {
    private static final JsonMapper JSON = new JsonMapper();

    @Test
    @Timeout(90)
    void aprioriAndFpGrowthRunInsideAssociatePanel() throws Exception {
        for (String className : new String[]{"weka.associations.Apriori", "weka.associations.FPGrowth"}) {
            Explorer explorer = new Explorer();
            explorer.getPreprocessPanel().setInstances(TestSupport.weather());
            SwingUtilities.invokeAndWait(() -> { });
            ExplorerSession session = new ExplorerSession(explorer);
            try {
                ObjectNode arguments = JSON.createObjectNode();
                arguments.put("associator_class", className);
                arguments.put("num_rules", 10);
                session.call(ExplorerSession.ASSOCIATE, arguments);
                JsonNode result = session.call(ExplorerSession.ASSOCIATE_RESULTS,
                        JSON.createObjectNode().put("wait_seconds", 30));
                assertTrue(result.get("complete").booleanValue(), className + ": " + result);
                assertTrue(result.get("output").asText().contains("play"), className);
                assertFalse(result.get("associator_running").booleanValue(), className);
                assertEquals("COMPLETE", result.get("result_status").asText());
                assertFalse(result.get("dataset_changed_since_training").booleanValue());
                assertFalse(result.get("operation_id").asText().isBlank());
                JsonNode state = session.call(ExplorerSession.STATE, null);
                assertEquals("Associate", state.get("active_tab").asText());
                assertEquals(1, state.get("association_result_count").intValue());
                assertEquals(5, state.get("attributes").intValue());
                if (className.endsWith("FPGrowth")) {
                    assertEquals("panel_only_binary_nominal", state.get("association_data_scope").asText());
                    assertTrue(state.get("association_attributes").intValue() > 5);
                }
                assertFalse(state.get("task_running").booleanValue());
                assertFalse(session.call(ExplorerSession.ASSOCIATE_STOP, null)
                        .get("stop_requested").booleanValue());
            } finally {
                session.close();
            }
        }
        assertEquals(21, ExplorerSession.toolSpecifications().size());
    }

    @Test
    void associationRequiresNominalAttributes() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> session.call(ExplorerSession.ASSOCIATE, JSON.createObjectNode()));
        assertTrue(error.getMessage().contains("Discretize"));
        assertFalse(session.call(ExplorerSession.STATE, null)
                .get("associator_running").booleanValue());
    }
}
