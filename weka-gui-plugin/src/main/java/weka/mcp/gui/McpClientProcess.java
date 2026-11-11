package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public final class McpClientProcess implements AutoCloseable {
    private static final JsonMapper JSON = new JsonMapper();
    private final Path serviceJar;
    private final Path configFile;
    private final Object requestLock = new Object();
    private final StringBuilder stderr = new StringBuilder();
    private Process process;
    private BufferedWriter input;
    private BufferedReader output;
    private ExecutorService responseReader;
    private ExecutorService errorReader;
    private long sequence;
    private boolean initialized;

    public McpClientProcess(Path serviceJar, Path configFile) {
        this.serviceJar = serviceJar.toAbsolutePath().normalize();
        this.configFile = configFile.toAbsolutePath().normalize();
    }

    public void start() throws Exception {
        synchronized (requestLock) {
            if (isRunning()) {
                return;
            }
            String javaExecutable = Path.of(System.getProperty("java.home"), "bin",
                    System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows")
                            ? "java.exe" : "java").toString();
            List<String> command = new ArrayList<>(List.of(
                    javaExecutable, "-Dfile.encoding=UTF-8", "-Djava.awt.headless=true",
                    "-Dweka.mcp.config=" + configFile, "-jar", serviceJar.toString()));
            process = new ProcessBuilder(command).start();
            input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            responseReader = Executors.newSingleThreadExecutor(runnable -> daemon(runnable, "mcp-gui-response"));
            errorReader = Executors.newSingleThreadExecutor(runnable -> daemon(runnable, "mcp-gui-stderr"));
            errorReader.execute(this::drainStderr);
            try {
                awaitStartup();
                request("initialize", Map.of(
                        "protocolVersion", "2025-06-18",
                        "capabilities", Map.of(),
                        "clientInfo", Map.of("name", "weka-mcp-gui", "version", "0.8.0")), 30);
                initialized = true;
                ObjectNode notification = JSON.createObjectNode();
                notification.put("jsonrpc", "2.0");
                notification.put("method", "notifications/initialized");
                notification.set("params", JSON.createObjectNode());
                writeMessage(notification);
            } catch (Exception error) {
                abort();
                close();
                throw error;
            }
        }
    }

    public JsonNode callTool(String name, Map<String, Object> arguments) throws Exception {
        return request("tools/call", Map.of("name", name, "arguments", arguments == null ? Map.of() : arguments), 150);
    }

    public JsonNode request(String method, Object parameters, long timeoutSeconds) throws Exception {
        synchronized (requestLock) {
            if (process == null || !process.isAlive()) {
                throw new McpException("The MCP process is not running");
            }
            long id = ++sequence;
            ObjectNode request = JSON.createObjectNode();
            request.put("jsonrpc", "2.0");
            request.put("id", id);
            request.put("method", method);
            request.set("params", JSON.valueToTree(parameters));
            writeMessage(request);
            JsonNode response = readResponse(id, timeoutSeconds);
            if (response.has("error")) {
                throw new McpException(response.get("error").toString());
            }
            JsonNode result = response.get("result");
            if (result != null && result.path("isError").booleanValue(false)) {
                String text = result.path("content").path(0).path("text").asText("Unknown MCP tool error");
                throw new McpException(text);
            }
            if (result != null && "text".equals(result.path("content").path(0).path("type").asText())) {
                return JSON.readTree(result.path("content").path(0).path("text").asText());
            }
            return result;
        }
    }

    private JsonNode readResponse(long id, long timeoutSeconds) throws Exception {
        Future<String> future = responseReader.submit(output::readLine);
        try {
            String line = future.get(timeoutSeconds, TimeUnit.SECONDS);
            if (line == null) {
                throw new McpException("The MCP process closed stdout; " + stderr());
            }
            JsonNode response = JSON.readTree(line);
            if (!"2.0".equals(response.path("jsonrpc").asText()) || response.path("id").asLong() != id) {
                throw new McpException("Invalid JSON-RPC response: " + line);
            }
            return response;
        } catch (TimeoutException error) {
            future.cancel(true);
            abort();
            throw new McpException("MCP request timed out after " + timeoutSeconds + " seconds", error);
        } catch (InterruptedException error) {
            future.cancel(true);
            Thread.currentThread().interrupt();
            throw error;
        } catch (java.util.concurrent.ExecutionException error) {
            Throwable cause = error.getCause();
            if (cause instanceof Exception exception) {
                throw exception;
            }
            throw new McpException("Unable to read the MCP response", error);
        }
    }

    private void writeMessage(JsonNode message) throws IOException {
        input.write(JSON.writeValueAsString(message));
        input.newLine();
        input.flush();
    }

    private void awaitStartup() throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
        while (System.nanoTime() < deadline) {
            if (stderr().contains("started with 43 tools")) {
                return;
            }
            if (!process.isAlive()) {
                throw new McpException("MCP exited during startup with code " + process.exitValue() + "; " + stderr());
            }
            Thread.sleep(25);
        }
        throw new McpException("MCP startup timed out; " + stderr());
    }

    private void drainStderr() {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                synchronized (stderr) {
                    stderr.append(line).append('\n');
                }
            }
        } catch (IOException ignored) {
        }
    }

    public Integer exitCode() {
        return process == null || process.isAlive() ? null : process.exitValue();
    }

    public boolean isRunning() {
        return initialized && process != null && process.isAlive();
    }

    public String stderr() {
        synchronized (stderr) {
            return stderr.toString();
        }
    }

    public void abort() {
        if (process != null) {
            process.destroy();
        }
    }

    @Override
    public void close() {
        try {
            if (input != null) {
                input.close();
            }
        } catch (IOException ignored) {
        }
        if (process != null) {
            try {
                if (!process.waitFor(15, TimeUnit.SECONDS)) {
                    process.destroy();
                    if (!process.waitFor(5, TimeUnit.SECONDS)) {
                        process.destroyForcibly();
                    }
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
        }
        if (responseReader != null) {
            responseReader.shutdownNow();
        }
        if (errorReader != null) {
            errorReader.shutdownNow();
        }
        initialized = false;
    }

    private static Thread daemon(Runnable runnable, String name) {
        Thread thread = new Thread(runnable, name);
        thread.setDaemon(true);
        return thread;
    }

    public static final class McpException extends Exception {
        public McpException(String message) {
            super(message);
        }

        public McpException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
