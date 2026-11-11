package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.WekaPackageManager;
import weka.gui.explorer.Explorer;

import javax.swing.AbstractButton;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class NativeBridgeIT {
    private static final JsonMapper JSON = new JsonMapper();
    private Process connector;
    private BufferedWriter input;
    private BufferedReader output;
    private int sequence;

    @Test
    @Timeout(180)
    void externalMcpClientDrivesRealExplorerAndExistingAnalysisService() throws Exception {
        Path bridgeFile = Path.of("target/native-agent-bridge.json").toAbsolutePath();
        String previousFile = System.getProperty("weka.mcp.gui.bridge.file");
        System.setProperty("weka.mcp.gui.bridge.file", bridgeFile.toString());
        if (!Files.isRegularFile(WekaPackageManager.PACKAGES_DIR.toPath()
                .resolve("wekaMcp/weka-mcp-gui-0.8.0.jar"))) {
            WekaPackageManager.installPackageFromArchive("target/weka-mcp-gui-0.8.0.zip");
        }
        WekaPackageManager.loadPackages(false, true, false);
        TestSupport.iris();
        JFrame frame = new JFrame("Weka native MCP bridge verification");
        try {
            Explorer explorer = new Explorer();
            explorer.getPreprocessPanel().setInstances(TestSupport.iris());
            SwingUtilities.invokeAndWait(() -> { });
            McpPanel panel = find(explorer, McpPanel.class, null);
            assertNotNull(panel);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());
            frame.add(explorer, BorderLayout.CENTER);
            frame.setSize(1280, 900);
            SwingUtilities.invokeAndWait(() -> {
                frame.setVisible(true);
                explorer.getTabbedPane().setSelectedComponent(panel);
            });
            click(panel, "启动 Agent 桥接");
            assertTrue(Files.isRegularFile(bridgeFile));
            BridgeConnectionInfo connection = BridgeConnectionInfo.read(bridgeFile);
            if (Files.getFileStore(bridgeFile).supportsFileAttributeView("posix")) {
                Set<PosixFilePermission> permissions = Files.getPosixFilePermissions(bridgeFile);
                assertTrue(permissions.contains(PosixFilePermission.OWNER_READ));
                assertTrue(permissions.contains(PosixFilePermission.OWNER_WRITE));
                assertFalse(permissions.contains(PosixFilePermission.GROUP_READ));
                assertFalse(permissions.contains(PosixFilePermission.OTHERS_READ));
            }
            rejectInvalidToken(connection);
            startConnector(bridgeFile);
            ObjectNode initialization = JSON.createObjectNode();
            initialization.put("protocolVersion", "2025-06-18");
            initialization.set("capabilities", JSON.createObjectNode());
            ObjectNode clientInfo = initialization.putObject("clientInfo");
            clientInfo.put("name", "native-bridge-it");
            clientInfo.put("version", "1.0");
            assertEquals("weka-mcp-explorer", request("initialize", initialization)
                    .get("serverInfo").get("name").asText());
            ObjectNode notification = JSON.createObjectNode();
            notification.put("jsonrpc", "2.0");
            notification.put("method", "notifications/initialized");
            notification.set("params", JSON.createObjectNode());
            input.write(JSON.writeValueAsString(notification));
            input.newLine();
            input.flush();

            JsonNode tools = request("tools/list", JSON.createObjectNode()).get("tools");
            assertEquals(64, tools.size());
            assertTrue(tools.toString().contains(ExplorerSession.TRAIN));
            assertTrue(tools.toString().contains("weka_train_classifier"));
            JsonNode state = callTool(ExplorerSession.STATE, JSON.createObjectNode());
            assertEquals(30, state.get("instances").intValue());
            assertEquals("class", state.get("class_attribute").asText());
            JsonNode changed = callTool(ExplorerSession.SET_CLASS,
                    JSON.createObjectNode().put("class_index", 5));
            assertEquals("class", changed.get("class_attribute").asText());
            ObjectNode imported = JSON.createObjectNode();
            imported.put("path", "iris.arff");
            imported.put("class_index", 5);
            assertTrue(callTool(ExplorerSession.LOAD, imported).get("loaded").booleanValue());
            ObjectNode filtering = JSON.createObjectNode();
            filtering.put("filter_class", "weka.filters.unsupervised.attribute.ReplaceMissingValues");
            assertTrue(callTool(ExplorerSession.FILTER, filtering).get("filtered").booleanValue());
            ObjectNode profileArguments = JSON.createObjectNode();
            profileArguments.put("max_attributes", 5);
            profileArguments.put("max_nominal_values", 2);
            JsonNode profile = callTool(ExplorerSession.DESCRIBE, profileArguments);
            assertEquals(5, profile.get("attributes_returned").intValue());
            assertEquals(0, profile.get("missing_total").intValue());
            assertEquals(2, profile.get("attributes").get(4).get("values").size());
            assertTrue(profile.get("attributes").get(4).get("values_truncated").booleanValue());
            JsonNode undone = callTool(ExplorerSession.UNDO, JSON.createObjectNode());
            assertTrue(undone.get("changed").booleanValue());
            assertEquals(30, undone.get("instances").intValue());
            assertEquals(5, undone.get("class_index").intValue());
            ObjectNode training = JSON.createObjectNode();
            training.put("classifier_class", "weka.classifiers.trees.J48");
            training.put("folds", 3);
            training.put("seed", 1);
            assertEquals("RUNNING", callTool(ExplorerSession.TRAIN, training).get("status").asText());
            JsonNode result = callTool(ExplorerSession.RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(result.get("complete").booleanValue());
            assertTrue(result.get("output").asText().contains("Correctly Classified Instances"));
            assertFalse(result.get("classifier_running").booleanValue());
            assertEquals("COMPLETE", result.get("result_status").asText());
            assertFalse(result.get("dataset_changed_since_training").booleanValue());
            assertFalse(result.get("operation_id").asText().isBlank());
            assertEquals(30, result.get("evaluation_metrics").get("total_instances").intValue());
            assertTrue(result.get("evaluation_metrics").get("correctly_classified_percent").doubleValue() > 0);
            assertEquals("Classify", callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("active_tab").asText());
            var screenshot = new java.awt.image.BufferedImage(frame.getWidth(), frame.getHeight(),
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            SwingUtilities.invokeAndWait(() -> frame.paint(screenshot.getGraphics()));
            javax.imageio.ImageIO.write(screenshot, "png", Path.of("target/native-explorer-result.png").toFile());

            ObjectNode slowTraining = JSON.createObjectNode();
            slowTraining.put("classifier_class", "weka.classifiers.trees.RandomForest");
            slowTraining.put("folds", 2);
            slowTraining.putArray("options").add("-I").add("100000");
            callTool(ExplorerSession.TRAIN, slowTraining);
            assertTrue(callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("task_running").booleanValue());
            ObjectNode cluster = JSON.createObjectNode();
            cluster.put("clusterer_class", "weka.clusterers.SimpleKMeans");
            cluster.put("num_clusters", 3);
            cluster.put("seed", 1);
            ObjectNode busyRequest = JSON.createObjectNode();
            busyRequest.put("name", ExplorerSession.CLUSTER);
            busyRequest.set("arguments", cluster);
            JsonNode busy = request("tools/call", busyRequest);
            assertTrue(busy.path("isError").booleanValue());
            assertTrue(busy.toString().contains("Another native Explorer task is already running"));
            assertTrue(callTool(ExplorerSession.STOP, JSON.createObjectNode())
                    .get("stop_requested").booleanValue());
            JsonNode cancelled = callTool(ExplorerSession.RESULTS, JSON.createObjectNode().put("wait_seconds", 30));
            assertFalse(cancelled.get("classifier_running").booleanValue());
            assertEquals("CANCELLED", cancelled.get("result_status").asText());
            assertFalse(cancelled.get("complete").booleanValue());
            callTool(ExplorerSession.CLUSTER, cluster);
            JsonNode clustered = callTool(ExplorerSession.CLUSTER_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(clustered.get("complete").booleanValue());
            assertTrue(clustered.get("output").asText().contains("Clustered Instances"));
            assertFalse(clustered.get("clusterer_running").booleanValue());
            assertEquals("COMPLETE", clustered.get("result_status").asText());
            assertFalse(clustered.get("dataset_changed_since_training").booleanValue());
            assertTrue(clustered.get("evaluation_metrics").get("within_cluster_squared_error").doubleValue() > 0);
            assertEquals("Cluster", callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("active_tab").asText());
            var clusterScreenshot = new java.awt.image.BufferedImage(frame.getWidth(), frame.getHeight(),
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            SwingUtilities.invokeAndWait(() -> frame.paint(clusterScreenshot.getGraphics()));
            javax.imageio.ImageIO.write(clusterScreenshot, "png",
                    Path.of("target/native-cluster-result.png").toFile());
            assertTrue(callTool(ExplorerSession.UNDO, JSON.createObjectNode())
                    .get("existing_results_may_be_stale").booleanValue());
            ObjectNode discretize = JSON.createObjectNode();
            discretize.put("filter_class", "weka.filters.unsupervised.attribute.Discretize");
            callTool(ExplorerSession.FILTER, discretize);
            ObjectNode association = JSON.createObjectNode();
            association.put("associator_class", "weka.associations.FPGrowth");
            association.put("num_rules", 10);
            callTool(ExplorerSession.ASSOCIATE, association);
            JsonNode associations = callTool(ExplorerSession.ASSOCIATE_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(associations.get("complete").booleanValue());
            assertTrue(associations.get("output").asText().contains("Iris"));
            assertEquals("COMPLETE", associations.get("result_status").asText());
            assertFalse(associations.get("dataset_changed_since_training").booleanValue());
            JsonNode associationState = callTool(ExplorerSession.STATE, JSON.createObjectNode());
            assertEquals("Associate", associationState.get("active_tab").asText());
            assertEquals("panel_only_binary_nominal", associationState.get("association_data_scope").asText());
            assertEquals(5, associationState.get("attributes").intValue());
            assertTrue(associationState.get("association_attributes").intValue() > 5);
            var associationScreenshot = new java.awt.image.BufferedImage(frame.getWidth(), frame.getHeight(),
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            SwingUtilities.invokeAndWait(() -> frame.paint(associationScreenshot.getGraphics()));
            javax.imageio.ImageIO.write(associationScreenshot, "png",
                    Path.of("target/native-association-result.png").toFile());

            ObjectNode selection = JSON.createObjectNode();
            selection.put("evaluator_class", "weka.attributeSelection.GainRatioAttributeEval");
            selection.put("num_to_select", 3);
            selection.put("folds", 3);
            selection.put("seed", 1);
            callTool(ExplorerSession.SELECT_ATTRIBUTES, selection);
            JsonNode selected = callTool(ExplorerSession.SELECT_RESULTS,
                    JSON.createObjectNode().put("wait_seconds", 30));
            assertTrue(selected.get("complete").booleanValue());
            assertTrue(selected.get("output").asText().contains("petal"));
            assertEquals("COMPLETE", selected.get("result_status").asText());
            assertFalse(selected.get("dataset_changed_since_training").booleanValue());
            assertEquals("cross_validation", selected.get("evaluation_mode").asText());
            assertEquals("Select attributes", callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("active_tab").asText());
            var selectionScreenshot = new java.awt.image.BufferedImage(frame.getWidth(), frame.getHeight(),
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            SwingUtilities.invokeAndWait(() -> frame.paint(selectionScreenshot.getGraphics()));
            javax.imageio.ImageIO.write(selectionScreenshot, "png",
                    Path.of("target/native-attribute-selection-result.png").toFile());

            ObjectNode visualization = JSON.createObjectNode();
            visualization.put("x_attribute", "sepallength");
            visualization.put("y_attribute", "petalwidth");
            visualization.put("color_attribute", "class");
            visualization.put("width", 800);
            visualization.put("height", 600);
            assertTrue(callTool(ExplorerSession.VISUALIZE, visualization).get("displayed").booleanValue());
            assertTrue(callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("visualization_open").booleanValue());
            ObjectNode captureRequest = JSON.createObjectNode();
            captureRequest.put("name", ExplorerSession.CAPTURE_VISUALIZATION);
            captureRequest.set("arguments", visualization);
            JsonNode captured = request("tools/call", captureRequest);
            assertFalse(captured.path("isError").booleanValue(false));
            JsonNode imageContent = null;
            for (JsonNode content : captured.get("content")) {
                if (content.path("type").asText().equals("image")) {
                    imageContent = content;
                }
            }
            assertNotNull(imageContent, captured.toString());
            assertEquals("image/png", imageContent.get("mimeType").asText());
            byte[] png = java.util.Base64.getDecoder().decode(imageContent.get("data").asText());
            assertTrue(png.length > 8);
            var decoded = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(png));
            assertEquals(800, decoded.getWidth());
            assertEquals(600, decoded.getHeight());
            Files.write(Path.of("target/native-visualization.png"), png);
            assertTrue(Files.isRegularFile(Path.of(captured.get("structuredContent")
                    .get("image_path").asText())));
            assertTrue(callTool(ExplorerSession.CLOSE_VISUALIZATION, JSON.createObjectNode())
                    .get("closed").booleanValue());
            assertFalse(callTool(ExplorerSession.STATE, JSON.createObjectNode())
                    .get("visualization_open").booleanValue());

            ObjectNode load = JSON.createObjectNode();
            load.put("path", "iris.arff");
            load.put("class_index", 5);
            assertTrue(callTool("weka_load_dataset", load).get("dataset_id").isTextual());
            assertEquals(-32601, rawRequest("unknown/method", JSON.createObjectNode())
                    .get("error").get("code").asInt());
            input.close();
            if (!connector.waitFor(20, TimeUnit.SECONDS)) {
                connector.destroyForcibly();
                fail("STDIO connector did not close after stdin EOF");
            }
            assertEquals(0, connector.exitValue(), connector.getErrorStream().toString());
            click(panel, "停止 Agent 桥接");
            assertFalse(Files.exists(bridgeFile));
        } finally {
            SwingUtilities.invokeAndWait(frame::dispose);
            if (connector != null && connector.isAlive()) {
                connector.destroyForcibly();
            }
            if (previousFile == null) {
                System.clearProperty("weka.mcp.gui.bridge.file");
            } else {
                System.setProperty("weka.mcp.gui.bridge.file", previousFile);
            }
        }
    }

    private void startConnector(Path bridgeFile) throws Exception {
        Path packageDirectory = Path.of("target/package").toAbsolutePath();
        boolean windows = System.getProperty("os.name").toLowerCase().startsWith("windows");
        String launcher = packageDirectory.resolve(windows ? "bin/bridge-stdio.bat" : "bin/bridge-stdio.sh").toString();
        connector = windows
                ? new ProcessBuilder("cmd.exe", "/c", launcher, bridgeFile.toString()).start()
                : new ProcessBuilder(launcher, bridgeFile.toString()).start();
        input = new BufferedWriter(new OutputStreamWriter(connector.getOutputStream(), StandardCharsets.UTF_8));
        output = new BufferedReader(new InputStreamReader(connector.getInputStream(), StandardCharsets.UTF_8));
    }

    private JsonNode callTool(String name, ObjectNode arguments) throws Exception {
        ObjectNode parameters = JSON.createObjectNode();
        parameters.put("name", name);
        parameters.set("arguments", arguments);
        JsonNode result = request("tools/call", parameters);
        assertFalse(result.path("isError").booleanValue(false), result.toString());
        assertNotNull(result.get("structuredContent"), result.toString());
        return result.get("structuredContent");
    }

    private JsonNode request(String method, ObjectNode parameters) throws Exception {
        JsonNode response = rawRequest(method, parameters);
        assertFalse(response.has("error"), response.toString());
        return response.get("result");
    }

    private JsonNode rawRequest(String method, ObjectNode parameters) throws Exception {
        int id = ++sequence;
        ObjectNode request = JSON.createObjectNode();
        request.put("jsonrpc", "2.0");
        request.put("id", id);
        request.put("method", method);
        request.set("params", parameters);
        input.write(JSON.writeValueAsString(request));
        input.newLine();
        input.flush();
        var reader = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "native-bridge-test-reader");
            thread.setDaemon(true);
            return thread;
        });
        try {
            String line = reader.submit(output::readLine).get(45, TimeUnit.SECONDS);
            assertNotNull(line, "MCP bridge closed before responding");
            JsonNode response = JSON.readTree(line);
            assertEquals(id, response.get("id").asInt(), response.toString());
            return response;
        } finally {
            reader.shutdownNow();
        }
    }

    private static void rejectInvalidToken(BridgeConnectionInfo connection) throws Exception {
        try (Socket socket = new Socket(connection.host(), connection.port())) {
            socket.setSoTimeout(10_000);
            socket.getOutputStream().write("{\"token\":\"invalid\"}\n".getBytes(StandardCharsets.UTF_8));
            socket.getOutputStream().flush();
            BufferedReader response = new BufferedReader(new InputStreamReader(socket.getInputStream(),
                    StandardCharsets.UTF_8));
            JsonNode ready = JSON.readTree(response.readLine());
            assertFalse(ready.get("ready").booleanValue());
        }
    }

    private static void click(Container root, String text) throws Exception {
        AbstractButton button = find(root, AbstractButton.class, text);
        assertNotNull(button, text);
        SwingUtilities.invokeAndWait(button::doClick);
    }

    private static <T extends Component> T find(Container root, Class<T> type, String text) {
        for (Component component : root.getComponents()) {
            if (type.isInstance(component) && (text == null
                    || text.equals(component instanceof AbstractButton button ? button.getText() : null))) {
                return type.cast(component);
            }
            if (component instanceof Container container) {
                T result = find(container, type, text);
                if (result != null) {
                    return result;
                }
            }
        }
        return null;
    }
}
