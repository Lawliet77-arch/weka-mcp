package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.gui.explorer.Explorer;

import javax.swing.SwingUtilities;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExplorerVisualizationOperationsTest {
    private static final JsonMapper JSON = new JsonMapper();

    @Test
    @Timeout(60)
    void nativeScatterCanBeCapturedAndClosed() throws Exception {
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(TestSupport.iris());
        SwingUtilities.invokeAndWait(() -> { });
        ExplorerSession session = new ExplorerSession(explorer, TestSupport.workspace());
        try {
            ObjectNode arguments = JSON.createObjectNode();
            arguments.put("x_attribute", "petallength");
            arguments.put("y_attribute", "petalwidth");
            arguments.put("color_attribute", "class");
            arguments.put("width", 800);
            arguments.put("height", 600);
            JsonNode shown = session.call(ExplorerSession.VISUALIZE, arguments);
            assertTrue(shown.get("displayed").booleanValue());
            assertEquals("petallength", shown.get("x_attribute").asText());
            assertEquals("petalwidth", shown.get("y_attribute").asText());
            assertEquals("class", shown.get("color_attribute").asText());
            assertTrue(session.call(ExplorerSession.STATE, null)
                    .get("visualization_open").booleanValue());
            var captured = session.captureVisualization(arguments);
            assertEquals("image/png", captured.mimeType());
            assertTrue(captured.png().length > 8);
            var image = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(captured.png()));
            assertEquals(800, image.getWidth());
            assertEquals(600, image.getHeight());
            Path exported = Path.of(captured.metadata().get("image_path").asText());
            assertArrayEquals(captured.png(), Files.readAllBytes(exported));
            assertTrue(session.call(ExplorerSession.CLOSE_VISUALIZATION, null)
                    .get("closed").booleanValue());
            assertFalse(session.call(ExplorerSession.STATE, null)
                    .get("visualization_open").booleanValue());
            assertEquals(21, ExplorerSession.toolSpecifications().size());
        } finally {
            session.close();
        }
    }

    @Test
    void captureRequiresAnActiveNativeChart() {
        ExplorerSession session = new ExplorerSession(new Explorer());
        assertThrows(IllegalStateException.class,
                () -> session.captureVisualization(JSON.createObjectNode()));
    }
}
