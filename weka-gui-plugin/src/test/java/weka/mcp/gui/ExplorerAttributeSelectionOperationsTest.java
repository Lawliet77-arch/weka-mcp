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

class ExplorerAttributeSelectionOperationsTest {
    private static final JsonMapper JSON = new JsonMapper();

    @Test
    @Timeout(60)
    void cfsSubsetSelectionProducesNativeResult() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        try {
            JsonNode result = session.call(ExplorerSession.SELECT_ATTRIBUTES, JSON.createObjectNode());
            assertEquals("full_training_set", result.get("evaluation_mode").asText());
            JsonNode selected = session.call(ExplorerSession.SELECT_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(selected.get("complete").booleanValue(), selected.toString());
            assertEquals("COMPLETE", selected.get("result_status").asText());
            assertFalse(selected.get("dataset_changed_since_training").booleanValue());
            assertFalse(selected.get("operation_id").asText().isBlank());
            assertTrue(selected.get("output").asText().contains("petal"));
            assertTrue(selected.get("output").asText().contains("Selected attributes"));
            assertFalse(selected.get("attribute_selection_running").booleanValue());
            assertEquals("weka.attributeSelection.BestFirst", selected.get("search_class").asText());
            assertEquals("Select attributes", session.call(ExplorerSession.STATE, null)
                    .get("active_tab").asText());
        } finally {
            session.close();
        }
    }

    @Test
    @Timeout(60)
    void gainRatioRanksAttributesAndInfoGainUsesCrossValidation() throws Exception {
        for (String className : new String[]{"weka.attributeSelection.GainRatioAttributeEval",
                "weka.attributeSelection.InfoGainAttributeEval"}) {
            Explorer explorer = new Explorer();
            explorer.getPreprocessPanel().setInstances(TestSupport.iris());
            SwingUtilities.invokeAndWait(() -> { });
            ExplorerSession session = new ExplorerSession(explorer);
            try {
                ObjectNode arguments = JSON.createObjectNode();
                arguments.put("evaluator_class", className);
                arguments.put("num_to_select", 3);
                if (className.endsWith("InfoGainAttributeEval")) {
                    arguments.put("folds", 3);
                    arguments.put("seed", 1);
                }
                session.call(ExplorerSession.SELECT_ATTRIBUTES, arguments);
                JsonNode result = session.call(ExplorerSession.SELECT_RESULTS,
                        JSON.createObjectNode().put("wait_seconds", 30));
                assertTrue(result.get("complete").booleanValue(), className + ": " + result);
                assertTrue(result.get("output").asText().contains("petal"), className);
                assertEquals("weka.attributeSelection.Ranker", result.get("search_class").asText());
                assertEquals(3, result.get("num_to_select").intValue());
                if (className.endsWith("InfoGainAttributeEval")) {
                    assertEquals("cross_validation", result.get("evaluation_mode").asText());
                    assertEquals(3, result.get("folds").intValue());
                } else {
                    assertEquals("full_training_set", result.get("evaluation_mode").asText());
                }
                assertEquals(1, session.call(ExplorerSession.STATE, null)
                        .get("attribute_selection_result_count").intValue());
                assertFalse(session.call(ExplorerSession.SELECT_STOP, null)
                        .get("stop_requested").booleanValue());
            } finally {
                session.close();
            }
        }
        assertEquals(21, ExplorerSession.toolSpecifications().size());
    }

    @Test
    void selectionRequiresClassAttribute() throws Exception {
        Explorer explorer = new Explorer();
        var data = TestSupport.iris();
        data.setClassIndex(-1);
        explorer.getPreprocessPanel().setInstances(data);
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        assertThrows(IllegalStateException.class,
                () -> session.call(ExplorerSession.SELECT_ATTRIBUTES, JSON.createObjectNode()));
        assertFalse(session.call(ExplorerSession.STATE, null)
                .get("attribute_selection_running").booleanValue());
    }
}
