package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.time.Instant;
import java.util.Set;

public record BridgeConnectionInfo(String host, int port, String token) {
    private static final JsonMapper JSON = new JsonMapper();

    public static BridgeConnectionInfo read(Path path) throws IOException {
        JsonNode root = JSON.readTree(Files.readString(path));
        String host = required(root, "host");
        int port = root.path("port").intValue();
        String token = required(root, "token");
        try {
            if (!InetAddress.getByName(host).isLoopbackAddress()) {
                throw new IOException("Bridge host must be a loopback address");
            }
        } catch (IOException error) {
            throw error;
        } catch (Exception error) {
            throw new IOException("Invalid bridge host", error);
        }
        if (port < 1 || port > 65535 || token.length() < 32) {
            throw new IOException("Invalid bridge port or token");
        }
        return new BridgeConnectionInfo(host, port, token);
    }

    public static void write(Path destination, String host, int port, String token) throws IOException {
        Path absolute = destination.toAbsolutePath().normalize();
        Files.createDirectories(absolute.getParent());
        Path temporary = Files.createTempFile(absolute.getParent(), ".bridge-", ".json");
        try {
            restrict(temporary);
            ObjectNode root = JSON.createObjectNode();
            root.put("version", 1);
            root.put("host", host);
            root.put("port", port);
            root.put("token", token);
            root.put("pid", ProcessHandle.current().pid());
            root.put("created_at", Instant.now().toString());
            JSON.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), root);
            Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            restrict(absolute);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static String required(JsonNode root, String name) throws IOException {
        if (!root.path(name).isTextual() || root.path(name).asText().isBlank()) {
            throw new IOException("Missing bridge property: " + name);
        }
        return root.path(name).asText();
    }

    private static void restrict(Path path) throws IOException {
        try {
            Files.setPosixFilePermissions(path, Set.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE));
        } catch (UnsupportedOperationException ignored) {
        }
    }
}
