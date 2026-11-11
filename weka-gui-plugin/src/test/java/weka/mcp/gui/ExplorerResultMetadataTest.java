package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Instances;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExplorerResultMetadataTest {
    @Test
    void metricsAndDatasetProvenanceAreStructured() throws Exception {
        ObjectNode metrics = ExplorerResultMetadata.evaluationMetrics("""
                Correctly Classified Instances     28               93.3333 %
                Incorrectly Classified Instances     2                6.6667 %
                Kappa statistic                    0.9
                Mean absolute error                0.0444
                Root mean squared error            0.2108
                Relative absolute error            9.9567 %
                Root relative squared error       44.4865 %
                Total Number of Instances         30
                Within cluster sum of squared errors: 2.62833380459882
                """);
        assertEquals(28, metrics.get("correctly_classified").intValue());
        assertEquals(93.3333, metrics.get("correctly_classified_percent").doubleValue(), 0.00001);
        assertEquals(0.9, metrics.get("kappa").doubleValue(), 0.00001);
        assertEquals(0.2108, metrics.get("root_mean_squared_error").doubleValue(), 0.00001);
        assertEquals(9.9567, metrics.get("relative_absolute_error_percent").doubleValue(), 0.00001);
        assertEquals(30, metrics.get("total_instances").intValue());
        assertTrue(metrics.get("within_cluster_squared_error").doubleValue() > 2.6);

        Instances original = TestSupport.iris();
        ObjectNode snapshot = ExplorerResultMetadata.snapshot(original);
        ObjectNode result = new JsonMapper().createObjectNode();
        ExplorerResultMetadata.addProvenance(result, "operation-1", snapshot, false, true, original);
        assertEquals("operation-1", result.get("operation_id").asText());
        assertEquals("COMPLETE", result.get("result_status").asText());
        assertFalse(result.get("dataset_changed_since_training").booleanValue());
        assertNotEquals(ExplorerResultMetadata.version(original), ExplorerResultMetadata.version(new Instances(original)));
        ExplorerResultMetadata.addProvenance(result, "operation-1", snapshot, false, true, new Instances(original));
        assertEquals("STALE_DATASET", result.get("result_status").asText());
        assertTrue(result.get("dataset_changed_since_training").booleanValue());
    }
}
