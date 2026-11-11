package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.gui.explorer.Explorer;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public final class ExplorerBridgeServer implements AutoCloseable {
    private static final JsonMapper JSON = new JsonMapper();
    private static final int MAX_REQUEST_BYTES = 16 * 1024 * 1024;
    private final McpPanelController controller;
    private final PluginWorkspace workspace;
    private final Supplier<Explorer> explorer;
    private final Path connectionFile;
    private final SecureRandom random = new SecureRandom();
    private volatile boolean running;
    private volatile Socket client;
    private ServerSocket server;
    private Thread listener;
    private String token;
    private ExplorerSession nativeSession;

    public ExplorerBridgeServer(PluginWorkspace workspace, McpPanelController controller, Supplier<Explorer> explorer) {
        this.controller = controller;
        this.workspace = workspace;
        this.explorer = explorer;
        this.connectionFile = Path.of(System.getProperty("weka.mcp.gui.bridge.file",
                workspace.workspaceDirectory().resolve("bridge.json").toString()));
    }

    private synchronized ExplorerSession nativeSession() {
        if (nativeSession == null) {
            nativeSession = new ExplorerSession(explorer.get(), workspace);
        }
        return nativeSession;
    }

    public synchronized void start() throws IOException {
        if (isRunning()) {
            return;
        }
        byte[] secret = new byte[32];
        random.nextBytes(secret);
        token = Base64.getUrlEncoder().withoutPadding().encodeToString(secret);
        InetAddress loopback = InetAddress.getLoopbackAddress();
        server = new ServerSocket();
        try {
            server.bind(new java.net.InetSocketAddress(loopback, 0));
            BridgeConnectionInfo.write(connectionFile, server.getInetAddress().getHostAddress(),
                    server.getLocalPort(), token);
            running = true;
            listener = new Thread(this::accept, "weka-mcp-native-bridge");
            listener.setDaemon(true);
            listener.start();
        } catch (IOException error) {
            running = false;
            server.close();
            throw error;
        }
    }

    public boolean isRunning() {
        return running && server != null && !server.isClosed();
    }

    public Path connectionFile() {
        return connectionFile;
    }

    private void accept() {
        while (isRunning()) {
            try {
                Socket accepted = server.accept();
                client = accepted;
                try {
                    serve(accepted);
                } finally {
                    accepted.close();
                    client = null;
                }
            } catch (IOException error) {
                if (isRunning()) {
                    controller.log("Native bridge connection failed: " + error.getMessage());
                }
            } catch (Exception error) {
                if (isRunning()) {
                    controller.log("Native bridge request failed: " + error.getMessage());
                }
            }
        }
    }

    private void serve(Socket socket) throws Exception {
        socket.setSoTimeout(10_000);
        String handshake = readLine(socket.getInputStream(), 8192);
        if (handshake == null) {
            return;
        }
        JsonNode request = JSON.readTree(handshake);
        byte[] provided = request.path("token").asText().getBytes(StandardCharsets.UTF_8);
        if (!MessageDigest.isEqual(token.getBytes(StandardCharsets.UTF_8), provided)) {
            write(socket.getOutputStream(), error("Authentication failed"));
            return;
        }
        ObjectNode ready = JSON.createObjectNode();
        ready.put("ready", true);
        write(socket.getOutputStream(), ready);
        socket.setSoTimeout(0);
        while (isRunning()) {
            String line = readLine(socket.getInputStream(), MAX_REQUEST_BYTES);
            if (line == null) {
                return;
            }
            ObjectNode response = handleLine(line);
            if (response != null) {
                write(socket.getOutputStream(), response);
            }
        }
    }

    private ObjectNode handleLine(String line) {
        JsonNode id = null;
        try {
            JsonNode request = JSON.readTree(line);
            if (request == null || !request.isObject() || !"2.0".equals(request.path("jsonrpc").asText())) {
                return protocolError(JSON.nullNode(), -32600, "Invalid JSON-RPC request");
            }
            id = request.get("id");
            String method = request.path("method").asText();
            if (method.startsWith("notifications/")) {
                return null;
            }
            if (id == null || id.isNull()) {
                return protocolError(JSON.nullNode(), -32600, "A request id is required");
            }
            JsonNode result = dispatch(method, request.path("params"));
            ObjectNode response = JSON.createObjectNode();
            response.put("jsonrpc", "2.0");
            response.set("id", id);
            response.set("result", result);
            return response;
        } catch (UnknownMethod error) {
            return protocolError(id == null ? JSON.nullNode() : id, -32601, error.getMessage());
        } catch (Exception error) {
            return protocolError(id == null ? JSON.nullNode() : id, -32603,
                    error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage());
        }
    }

    private JsonNode dispatch(String method, JsonNode parameters) throws Exception {
        return switch (method) {
            case "initialize" -> initialize();
            case "ping" -> JSON.createObjectNode();
            case "tools/list" -> tools(parameters);
            case "tools/call" -> call(parameters);
            default -> throw new UnknownMethod("Method not supported: " + method);
        };
    }

    private ObjectNode initialize() {
        ObjectNode result = JSON.createObjectNode();
        result.put("protocolVersion", "2025-06-18");
        ObjectNode capabilities = JSON.createObjectNode();
        ObjectNode tools = JSON.createObjectNode();
        tools.put("listChanged", false);
        capabilities.set("tools", tools);
        result.set("capabilities", capabilities);
        ObjectNode serverInfo = JSON.createObjectNode();
        serverInfo.put("name", "weka-mcp-explorer");
        serverInfo.put("version", "0.8.0");
        result.set("serverInfo", serverInfo);
        result.put("instructions", "Native Explorer tools operate on the attached Weka window. "
                + "Use weka_explorer_get_results with wait_seconds after starting a training task. "
                + "Only one local bridge client is served at a time; close the connector when finished.");
        return result;
    }

    private JsonNode tools(JsonNode parameters) throws Exception {
        JsonNode core = coreRequest("tools/list", parameters, 30);
        ObjectNode result = (ObjectNode) core.deepCopy();
        ArrayNode tools = result.withArray("tools");
        ExplorerSession.toolSpecifications().forEach(tools::add);
        return result;
    }

    private JsonNode call(JsonNode parameters) throws Exception {
        String name = parameters.path("name").asText();
        if (name.isBlank()) {
            throw new IllegalArgumentException("Tool name is required");
        }
        JsonNode arguments = parameters.path("arguments");
        try {
            if (name.equals(ExplorerSession.CAPTURE_VISUALIZATION)) {
                return imageResult(nativeSession().captureVisualization(arguments));
            }
            if (name.startsWith("weka_explorer_")) {
                return toolResult(nativeSession().call(name, arguments));
            }
            ObjectNode request = JSON.createObjectNode();
            request.put("name", name);
            request.set("arguments", arguments.isMissingNode() || arguments.isNull()
                    ? JSON.createObjectNode() : arguments);
            JsonNode result = coreRequest("tools/call", request, 150);
            return toolResult(result);
        } catch (Exception error) {
            ObjectNode result = JSON.createObjectNode();
            ArrayNode content = result.putArray("content");
            ObjectNode text = content.addObject();
            text.put("type", "text");
            text.put("text", error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage());
            result.put("isError", true);
            return result;
        }
    }

    private static ObjectNode imageResult(ExplorerVisualizationOperations.Capture capture) {
        ObjectNode result = JSON.createObjectNode();
        ArrayNode content = result.putArray("content");
        ObjectNode text = content.addObject();
        text.put("type", "text");
        text.put("text", capture.metadata().toString());
        ObjectNode image = content.addObject();
        image.put("type", "image");
        image.put("data", java.util.Base64.getEncoder().encodeToString(capture.png()));
        image.put("mimeType", capture.mimeType());
        result.set("structuredContent", capture.metadata());
        result.put("isError", false);
        return result;
    }

    private static ObjectNode toolResult(JsonNode payload) {
        ObjectNode result = JSON.createObjectNode();
        ArrayNode content = result.putArray("content");
        ObjectNode text = content.addObject();
        text.put("type", "text");
        text.put("text", payload == null ? "null" : payload.toString());
        if (payload != null && payload.isObject()) {
            result.set("structuredContent", payload);
        }
        result.put("isError", false);
        return result;
    }

    private JsonNode coreRequest(String method, JsonNode parameters, long timeout) throws Exception {
        ensureCore();
        return controller.request(method, parameters, timeout);
    }

    private void ensureCore() throws Exception {
        if (controller.isRunning()) {
            return;
        }
        CountDownLatch started = new CountDownLatch(1);
        AtomicReference<Exception> failure = new AtomicReference<>();
        controller.start(started::countDown, error -> {
            failure.set(error);
            started.countDown();
        });
        if (!started.await(35, TimeUnit.SECONDS)) {
            throw new IOException("Timed out starting the MCP service");
        }
        if (failure.get() != null) {
            throw failure.get();
        }
        if (!controller.isRunning()) {
            throw new IOException("MCP service is not running");
        }
    }

    private static ObjectNode error(String message) {
        ObjectNode result = JSON.createObjectNode();
        result.put("ready", false);
        result.put("error", message);
        return result;
    }

    private static ObjectNode protocolError(JsonNode id, int code, String message) {
        ObjectNode response = JSON.createObjectNode();
        response.put("jsonrpc", "2.0");
        response.set("id", id);
        ObjectNode error = response.putObject("error");
        error.put("code", code);
        error.put("message", message);
        return response;
    }

    private static void write(OutputStream output, JsonNode message) throws IOException {
        output.write(JSON.writeValueAsString(message).getBytes(StandardCharsets.UTF_8));
        output.write('\n');
        output.flush();
    }

    private static String readLine(InputStream input, int maximum) throws IOException {
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        while (true) {
            int next = input.read();
            if (next < 0) {
                return buffer.size() == 0 ? null : buffer.toString(StandardCharsets.UTF_8);
            }
            if (next == '\n') {
                return buffer.toString(StandardCharsets.UTF_8);
            }
            if (next != '\r') {
                buffer.write(next);
            }
            if (buffer.size() > maximum) {
                throw new EOFException("MCP request exceeds " + maximum + " bytes");
            }
        }
    }

    @Override
    public void close() {
        running = false;
        try {
            if (server != null) {
                server.close();
            }
            if (client != null) {
                client.close();
            }
            if (explorer.get() != null) {
                nativeSession().close();
            }
            if (listener != null) {
                listener.join(2000);
            }
        } catch (Exception ignored) {
        }
        if (token != null) {
            try {
                if (Files.isRegularFile(connectionFile)
                        && token.equals(BridgeConnectionInfo.read(connectionFile).token())) {
                    Files.deleteIfExists(connectionFile);
                }
            } catch (IOException ignored) {
            }
        }
    }

    private static final class UnknownMethod extends Exception {
        private UnknownMethod(String message) {
            super(message);
        }
    }
}
