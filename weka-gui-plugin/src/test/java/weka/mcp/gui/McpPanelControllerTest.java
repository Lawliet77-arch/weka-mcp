package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import weka.gui.explorer.Explorer;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class McpPanelControllerTest {
    @Test
    void operationsUpdateStateAndReturnOnEdt() throws Exception {
        PluginWorkspace workspace = TestSupport.workspace();
        TestSupport.iris();
        McpPanelController controller = new McpPanelController(workspace);
        try {
            CountDownLatch started = new CountDownLatch(1);
            controller.start(started::countDown, error -> {
                throw new AssertionError(error);
            });
            assertTrue(started.await(30, TimeUnit.SECONDS));
            assertTrue(controller.isRunning());
            CountDownLatch loaded = new CountDownLatch(1);
            AtomicReference<JsonNode> dataset = new AtomicReference<>();
            AtomicReference<Boolean> callbackOnEdt = new AtomicReference<>(false);
            controller.callTool("weka_load_dataset", Map.of("path", "iris.arff", "class_index", 5), result -> {
                callbackOnEdt.set(javax.swing.SwingUtilities.isEventDispatchThread());
                dataset.set(result);
                loaded.countDown();
            }, error -> {
                throw new AssertionError(error);
            });
            assertTrue(loaded.await(30, TimeUnit.SECONDS));
            assertTrue(callbackOnEdt.get());
            assertEquals(30, dataset.get().get("instances").intValue());
            CountDownLatch stopped = new CountDownLatch(1);
            controller.stop(stopped::countDown, error -> {
                throw new AssertionError(error);
            });
            assertTrue(stopped.await(20, TimeUnit.SECONDS));
            assertFalse(controller.isRunning());
            assertFalse(controller.isBusy());
        } finally {
            controller.close();
        }
    }

    @Test
    void panelImplementsExplorerContract() throws Exception {
        PluginWorkspace workspace = TestSupport.workspace();
        McpPanel panel = new McpPanel(workspace);
        weka.gui.explorer.Explorer explorer = new weka.gui.explorer.Explorer();
        panel.setExplorer(explorer);
        assertEquals(explorer, panel.getExplorer());
        assertEquals("Weka MCP", panel.getTabTitle());
        assertFalse(panel.getTabTitleToolTip().isBlank());
        panel.setInstances(TestSupport.iris());
        panel.setLog(new weka.gui.Logger() {
            @Override
            public void logMessage(String message) {
            }

            @Override
            public void statusMessage(String message) {
            }
        });
        panel.close();
        panel.removeNotify();
        assertTrue(panel instanceof Explorer.ExplorerPanel);
    }
}
