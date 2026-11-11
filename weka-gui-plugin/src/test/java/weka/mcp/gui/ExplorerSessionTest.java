package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ObjectNode;
import weka.gui.explorer.Explorer;

import javax.swing.SwingUtilities;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExplorerSessionTest {
    @Test
    @Timeout(60)
    void nativeSessionTrainsInsideRealExplorerAndReturnsItsResult() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer, TestSupport.workspace());
        try {
            JsonNode state = session.call(ExplorerSession.STATE, null);
            assertEquals(30, state.get("instances").intValue());
            assertEquals("class", state.get("class_attribute").asText());
            ObjectNode imported = new tools.jackson.databind.json.JsonMapper().createObjectNode();
            imported.put("path", java.nio.file.Path.of("src/test/resources/iris.arff").toAbsolutePath().toString());
            assertEquals(30, session.call(ExplorerSession.LOAD, imported).get("instances").intValue());
            ObjectNode invalidClass = (ObjectNode) imported.deepCopy();
            invalidClass.put("class_index", "last");
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                    () -> session.call(ExplorerSession.LOAD, invalidClass));
            ObjectNode profileArguments = new tools.jackson.databind.json.JsonMapper().createObjectNode();
            profileArguments.put("max_attributes", 5);
            profileArguments.put("max_nominal_values", 2);
            JsonNode profile = session.call(ExplorerSession.DESCRIBE, profileArguments);
            assertEquals(5, profile.get("attributes_returned").intValue());
            assertEquals(0, profile.get("missing_total").intValue());
            assertTrue(profile.get("attributes").get(0).has("mean"));
            assertEquals(2, profile.get("attributes").get(4).get("values").size());
            assertTrue(profile.get("attributes").get(4).get("values_truncated").booleanValue());
            double originalMinimum = explorer.getPreprocessPanel().getInstances()
                    .attributeStats(0).numericStats.min;
            ObjectNode filtering = new tools.jackson.databind.json.JsonMapper().createObjectNode();
            filtering.put("filter_class", "weka.filters.unsupervised.attribute.ReplaceMissingValues");
            JsonNode filtered = session.call(ExplorerSession.FILTER, filtering);
            assertTrue(filtered.get("filtered").booleanValue());
            assertEquals(30, filtered.get("instances").intValue());
            assertTrue(filtered.get("undo_available").booleanValue());
            filtering.put("filter_class", "weka.filters.unsupervised.attribute.Normalize");
            filtering.putArray("options").add("-S").add("1.0").add("-T").add("0.0");
            session.call(ExplorerSession.FILTER, filtering);
            ObjectNode firstAttribute = new tools.jackson.databind.json.JsonMapper().createObjectNode();
            firstAttribute.put("max_attributes", 1);
            JsonNode normalized = session.call(ExplorerSession.DESCRIBE, firstAttribute);
            assertEquals(0.0, normalized.get("attributes").get(0).get("min").doubleValue(), 0.000001);
            assertEquals(1.0, normalized.get("attributes").get(0).get("max").doubleValue(), 0.000001);
            JsonNode undone = session.call(ExplorerSession.UNDO, null);
            assertTrue(undone.get("changed").booleanValue());
            assertEquals(5, undone.get("class_index").intValue());
            JsonNode restored = session.call(ExplorerSession.DESCRIBE, firstAttribute);
            assertEquals(originalMinimum, restored.get("attributes").get(0).get("min").doubleValue(), 0.000001);
            JsonNode changed = session.call(ExplorerSession.SET_CLASS,
                    sessionArguments("class_index", 5));
            assertEquals(5, changed.get("class_index").intValue());
            JsonNode started = session.call(ExplorerSession.TRAIN, new tools.jackson.databind.json.JsonMapper()
                    .valueToTree(Map.of("classifier_class", "weka.classifiers.trees.J48", "folds", 3, "seed", 1)));
            assertEquals("RUNNING", started.get("status").asText());
            JsonNode result = session.call(ExplorerSession.RESULTS, sessionArguments("wait_seconds", 30));
            assertTrue(result.get("complete").booleanValue());
            assertFalse(result.get("classifier_running").booleanValue());
            assertEquals("COMPLETE", result.get("result_status").asText());
            assertFalse(result.get("dataset_changed_since_training").booleanValue());
            assertFalse(result.get("operation_id").asText().isBlank());
            assertEquals(30, result.get("evaluation_metrics").get("total_instances").intValue());
            assertTrue(result.get("evaluation_metrics").get("correctly_classified_percent").doubleValue() > 0);
            assertEquals(result.get("training_dataset").get("dataset_version").asText(),
                    session.call(ExplorerSession.STATE, null).get("dataset_version").asText());
            assertTrue(result.get("output").asText().contains("Correctly Classified Instances"));
            assertTrue(result.get("output").asText().contains("3"));
            assertTrue(session.call(ExplorerSession.STATE, null).get("active_tab").asText()
                    .contains("Classify"));
            assertFalse(session.call(ExplorerSession.STOP, null).get("stop_requested").booleanValue());
            assertEquals(21, ExplorerSession.toolSpecifications().size());
        } finally {
            session.close();
        }
    }

    @Test
    @Timeout(60)
    void completedClassifierResultBecomesStaleAfterDatasetChange() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer);
        try {
            session.call(ExplorerSession.TRAIN, new tools.jackson.databind.json.JsonMapper()
                    .valueToTree(Map.of("classifier_class", "weka.classifiers.trees.J48", "folds", 3, "seed", 1)));
            JsonNode completed = session.call(ExplorerSession.RESULTS, sessionArguments("wait_seconds", 30));
            assertEquals("COMPLETE", completed.get("result_status").asText());
            ObjectNode filtering = new tools.jackson.databind.json.JsonMapper().createObjectNode();
            filtering.put("filter_class", "weka.filters.unsupervised.attribute.ReplaceMissingValues");
            session.call(ExplorerSession.FILTER, filtering);
            JsonNode stale = session.call(ExplorerSession.RESULTS, null);
            assertEquals("STALE_DATASET", stale.get("result_status").asText());
            assertTrue(stale.get("dataset_changed_since_training").booleanValue());
            assertEquals(completed.get("operation_id"), stale.get("operation_id"));
            org.junit.jupiter.api.Assertions.assertNotEquals(
                    stale.get("training_dataset").get("dataset_version").asText(),
                    stale.get("current_dataset").get("dataset_version").asText());
        } finally {
            session.close();
        }
    }

    @Test
    void undoWithoutHistoryLeavesExplorerUnchanged() throws Exception {
        ExplorerSession session = new ExplorerSession(new Explorer());
        JsonNode result = session.call(ExplorerSession.UNDO, null);
        assertFalse(result.get("changed").booleanValue());
        assertFalse(result.get("dataset_loaded").booleanValue());
        assertEquals("preprocess_dataset", result.get("scope").asText());
    }

    private static JsonNode sessionArguments(String name, int value) {
        return new tools.jackson.databind.json.JsonMapper().createObjectNode().put(name, value);
    }
}
