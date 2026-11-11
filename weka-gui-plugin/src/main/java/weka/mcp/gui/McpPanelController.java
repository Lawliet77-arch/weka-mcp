package weka.mcp.gui;

import tools.jackson.databind.JsonNode;

import javax.swing.SwingUtilities;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public final class McpPanelController implements AutoCloseable {
    private final PluginWorkspace workspace;
    private final ExecutorService tasks;
    private final AtomicBoolean busy = new AtomicBoolean();
    private volatile McpClientProcess client;
    private volatile Consumer<String> logger = ignored -> { };

    public McpPanelController(PluginWorkspace workspace) {
        this.workspace = workspace;
        this.tasks = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "weka-mcp-gui-worker");
            thread.setDaemon(true);
            return thread;
        });
    }

    public void start(Runnable success, Consumer<Exception> error) {
        if (isRunning()) {
            SwingUtilities.invokeLater(success);
            return;
        }
        execute(() -> {
            McpClientProcess created = workspace.newClient();
            client = created;
            created.start();
            return null;
        }, ignored -> success.run(), error);
    }

    public void callTool(String tool, Map<String, Object> arguments,
            java.util.function.Consumer<JsonNode> success, Consumer<Exception> error) {
        execute(() -> callTool(tool, arguments), success, error);
    }

    public JsonNode callTool(String tool, Map<String, Object> arguments) throws Exception {
        McpClientProcess current = client;
        if (current == null || !current.isRunning()) {
            throw new McpClientProcess.McpException("Start the MCP service first");
        }
        return current.callTool(tool, arguments);
    }

    public void execute(Callable<JsonNode> action, java.util.function.Consumer<JsonNode> success,
            Consumer<Exception> error) {
        if (!busy.compareAndSet(false, true)) {
            error.accept(new IllegalStateException("Another Weka MCP operation is still running"));
            return;
        }
        tasks.execute(() -> {
            try {
                JsonNode result = action.call();
                busy.set(false);
                if (success != null) {
                    SwingUtilities.invokeLater(() -> {
                        try {
                            success.accept(result);
                        } catch (RuntimeException exception) {
                            error.accept(exception);
                        }
                    });
                }
            } catch (Exception exception) {
                busy.set(false);
                SwingUtilities.invokeLater(() -> error.accept(exception));
            }
        });
    }

    public void stop(Runnable success, Consumer<Exception> error) {
        McpClientProcess current = client;
        if (current == null) {
            SwingUtilities.invokeLater(success);
            return;
        }
        current.abort();
        tasks.execute(() -> {
            try {
                current.close();
                client = null;
                SwingUtilities.invokeLater(success);
            } catch (Exception exception) {
                SwingUtilities.invokeLater(() -> error.accept(exception));
            } finally {
                busy.set(false);
            }
        });
    }

    public JsonNode request(String method, Map<String, Object> parameters, long timeoutSeconds) throws Exception {
        McpClientProcess current = client;
        if (current == null || !current.isRunning()) {
            throw new McpClientProcess.McpException("Start the MCP service first");
        }
        return current.request(method, parameters, timeoutSeconds);
    }

    public JsonNode request(String method, JsonNode parameters, long timeoutSeconds) throws Exception {
        McpClientProcess current = client;
        if (current == null || !current.isRunning()) {
            throw new McpClientProcess.McpException("Start the MCP service first");
        }
        return current.request(method, parameters, timeoutSeconds);
    }

    public void setLogger(Consumer<String> logger) {
        this.logger = logger == null ? ignored -> { } : logger;
    }

    public void log(String message) {
        logger.accept(message);
    }

    public boolean isBusy() {
        return busy.get();
    }

    public boolean isRunning() {
        McpClientProcess current = client;
        return current != null && current.isRunning();
    }

    public String stderr() {
        McpClientProcess current = client;
        return current == null ? "" : current.stderr();
    }

    @Override
    public void close() {
        McpClientProcess current = client;
        if (current != null) {
            current.abort();
            current.close();
            client = null;
        }
        tasks.shutdownNow();
    }
}
