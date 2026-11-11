package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public final class McpBridgeConnector {
    private static final JsonMapper JSON = new JsonMapper();

    private McpBridgeConnector() {
    }

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception error) {
            System.err.println("Unable to connect to the Weka Explorer bridge: " + error.getMessage());
            System.err.println("Open Explorer's Weka MCP tab and enable the Agent bridge first.");
            System.exit(1);
        }
    }

    private static void run(String[] args) throws Exception {
        Path connectionFile = connectionFile(args);
        BridgeConnectionInfo connection = BridgeConnectionInfo.read(connectionFile);
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getByName(connection.host()), connection.port()), 10_000);
            socket.setSoTimeout(10_000);
            ObjectNode authentication = JSON.createObjectNode();
            authentication.put("token", connection.token());
            socket.getOutputStream().write(JSON.writeValueAsString(authentication).getBytes(StandardCharsets.UTF_8));
            socket.getOutputStream().write('\n');
            socket.getOutputStream().flush();
            String readyLine = readLine(socket.getInputStream());
            JsonNode ready = readyLine == null ? null : JSON.readTree(readyLine);
            if (ready == null || !ready.path("ready").booleanValue(false)) {
                throw new IOException(ready == null ? "Bridge closed during authentication"
                        : ready.path("error").asText("Bridge authentication failed"));
            }
            socket.setSoTimeout(0);
            Thread input = new Thread(() -> forwardInput(socket), "weka-mcp-bridge-stdin");
            input.setDaemon(true);
            input.start();
            transfer(socket.getInputStream(), System.out);
            input.join(5000);
        }
    }

    private static void forwardInput(Socket socket) {
        try {
            transfer(System.in, socket.getOutputStream());
            socket.shutdownOutput();
        } catch (IOException ignored) {
        }
    }

    private static void transfer(InputStream input, OutputStream output) throws IOException {
        byte[] buffer = new byte[16 * 1024];
        int read;
        while ((read = input.read(buffer)) >= 0) {
            output.write(buffer, 0, read);
            output.flush();
        }
    }

    private static String readLine(InputStream input) throws IOException {
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
            if (buffer.size() > 8192) {
                throw new EOFException("Invalid bridge handshake");
            }
        }
    }

    private static Path connectionFile(String[] args) {
        if (args.length == 1 && !args[0].isBlank()) {
            return Path.of(args[0]);
        }
        if (args.length == 2 && "--bridge-file".equals(args[0])) {
            return Path.of(args[1]);
        }
        if (args.length > 0) {
            throw new IllegalArgumentException("Usage: McpBridgeConnector [--bridge-file] <bridge.json>");
        }
        return Path.of(System.getProperty("weka.mcp.gui.bridge.file",
                Path.of(System.getProperty("user.home"), ".weka-mcp-gui", "bridge.json").toString()));
    }
}
