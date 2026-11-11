package dev.wekamcp.engine;

import dev.wekamcp.config.ServerConfig;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TaskRunnerTest {
    @Test
    void timeoutRequestsCancellation() throws Exception {
        Path directory = Files.createTempDirectory("weka-task-test");
        Path configPath = directory.resolve("config.properties");
        Files.writeString(configPath, "dataRoot=data\nmaxConcurrentTasks=1\nmaxQueuedTasks=0\n");
        ServerConfig config = ServerConfig.load(configPath);
        AtomicBoolean interrupted = new AtomicBoolean();
        try (TaskRunner runner = new TaskRunner(config)) {
            assertThrows(IllegalStateException.class, () -> runner.call(1, () -> {
                try {
                    Thread.sleep(10_000);
                } catch (InterruptedException e) {
                    interrupted.set(true);
                    throw e;
                }
                return null;
            }));
            assertTrue(interrupted.get());
        } finally {
            Files.deleteIfExists(configPath);
            Files.deleteIfExists(directory.resolve("data"));
            Files.deleteIfExists(directory);
        }
    }
}
