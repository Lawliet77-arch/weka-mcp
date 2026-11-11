package dev.wekamcp.engine;

import dev.wekamcp.config.ServerConfig;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

public class TaskRunner implements AutoCloseable {
    private final ThreadPoolExecutor executor;

    public TaskRunner(ServerConfig config) {
        AtomicInteger sequence = new AtomicInteger();
        this.executor = new ThreadPoolExecutor(
                config.maxConcurrentTasks(),
                config.maxConcurrentTasks(),
                30L,
                TimeUnit.SECONDS,
                config.maxQueuedTasks() == 0
                        ? new java.util.concurrent.SynchronousQueue<>()
                        : new LinkedBlockingQueue<>(config.maxQueuedTasks()),
                runnable -> {
                    Thread thread = new Thread(runnable, "weka-task-" + sequence.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                });
    }

    public <T> T call(long timeoutSeconds, Callable<T> task) throws Exception {
        Future<T> future;
        try {
            future = executor.submit(task);
        } catch (RejectedExecutionException e) {
            throw new IllegalStateException("Too many Weka tasks are running or queued", e);
        }
        try {
            return future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw new IllegalStateException("Weka task exceeded " + timeoutSeconds
                    + " seconds; cancellation was requested but the algorithm may not stop immediately", e);
        } catch (InterruptedException e) {
            future.cancel(true);
            Thread.currentThread().interrupt();
            throw e;
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof Exception exception) {
                throw exception;
            }
            if (cause instanceof Error error) {
                throw error;
            }
            throw e;
        }
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}
