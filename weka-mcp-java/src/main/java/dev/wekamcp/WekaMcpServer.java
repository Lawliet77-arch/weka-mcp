package dev.wekamcp;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.tools.WekaTools;
import io.modelcontextprotocol.json.McpJsonDefaults;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.McpSyncServer;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.WekaPackageManager;

import java.io.FilterInputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

public final class WekaMcpServer {
    private static final Logger LOGGER = LoggerFactory.getLogger(WekaMcpServer.class);

    private WekaMcpServer() {
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("java.awt.headless", "true");
        PrintStream protocolOutput = System.out;
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8));

        ServerConfig config = ServerConfig.load();
        WekaPackageManager.loadPackages(false, true, false);
        Whitelist whitelist = Whitelist.load(config.whitelistPath());
        whitelist.selfCheck();
        SessionRegistry registry = new SessionRegistry(config);
        TaskRunner tasks = new TaskRunner(config);
        WekaTools tools = new WekaTools(config, whitelist, registry, tasks);
        CountDownLatch endOfInput = new CountDownLatch(1);
        StdioServerTransportProvider transport = new StdioServerTransportProvider(
                McpJsonDefaults.getMapper(), new EofTrackingInputStream(System.in, endOfInput), protocolOutput);
        McpSyncServer server = McpServer.sync(transport)
                .serverInfo("weka-mcp-java", "1.6.0")
                .capabilities(McpSchema.ServerCapabilities.builder().tools(false).build())
                .instructions("Use whitelisted Weka algorithms. Session UUIDs identify in-memory resources; experiment_id values support durable recovery. Import only trusted Java-serialized Weka models.")
                .tools(tools.specifications())
                .build();

        AtomicBoolean closed = new AtomicBoolean();
        Runnable cleanup = () -> {
            if (closed.compareAndSet(false, true)) {
                server.close();
                tasks.close();
            }
        };
        Thread shutdownHook = new Thread(() -> {
            endOfInput.countDown();
            cleanup.run();
        }, "weka-mcp-shutdown");
        Runtime.getRuntime().addShutdownHook(shutdownHook);
        LOGGER.info("weka-mcp-java 1.6.0 started with {} tools; dataRoot={}", tools.specifications().size(), config.dataRoot());
        try {
            endOfInput.await();
        } finally {
            cleanup.run();
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException ignored) {
            }
        }
    }

    private static final class EofTrackingInputStream extends FilterInputStream {
        private final CountDownLatch endOfInput;

        private EofTrackingInputStream(InputStream input, CountDownLatch endOfInput) {
            super(input);
            this.endOfInput = endOfInput;
        }

        @Override
        public int read() throws IOException {
            return track(super.read());
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            return track(super.read(buffer, offset, length));
        }

        private int track(int result) {
            if (result == -1) {
                endOfInput.countDown();
            }
            return result;
        }
    }
}
