package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import tools.jackson.databind.json.JsonMapper;
import weka.core.WekaPackageManager;
import weka.gui.explorer.Explorer;

import javax.swing.AbstractButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class McpPanelWorkflowIT {
    private JLabel status;

    @Test
    @Timeout(180)
    void registeredExplorerTabRunsRealWorkflowAndClosesItsProcess() throws Exception {
        assertFalse(Boolean.parseBoolean(System.getProperty("java.awt.headless")));
        if (!Files.isRegularFile(WekaPackageManager.PACKAGES_DIR.toPath()
                .resolve("wekaMcp/weka-mcp-gui-0.8.0.jar"))) {
            WekaPackageManager.installPackageFromArchive("target/weka-mcp-gui-0.8.0.zip");
        }
        WekaPackageManager.loadPackages(false, true, false);
        TestSupport.iris();
        JFrame frame = new JFrame("Weka MCP GUI integration verification");
        try {
            Explorer explorer = new Explorer();
            var instances = TestSupport.iris();
            instances.setClassIndex(instances.numAttributes() - 1);
            explorer.getPreprocessPanel().setInstances(instances);
            SwingUtilities.invokeAndWait(() -> { });
            McpPanel panel = find(explorer, McpPanel.class);
            assertNotNull(panel, "The Explorer must instantiate the registered Weka MCP panel");
            status = find(panel, JLabel.class, "MCP 未启动");
            assertNotNull(status);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());
            frame.add(explorer, BorderLayout.CENTER);
            frame.setSize(1280, 900);
            frame.setLocationRelativeTo(null);
            SwingUtilities.invokeAndWait(() -> {
                frame.setVisible(true);
                tabGroup(explorer).setSelectedComponent(panel);
            });
            assertTrue(onEdt(() -> panel.isShowing()));
            click(panel, "启动 MCP");
            awaitText(panel, "MCP 已就绪");
            click(panel, "加载与拆分");
            awaitText(panel, "数据已加载，训练/测试拆分完成");
            assertTrue(text(panel).contains("\"profile\""));
            click(panel, "训练与评估");
            awaitText(panel, "分类器训练与独立评估完成");
            String trained = text(panel);
            assertTrue(trained.contains("\"accuracy_percent\""));
            String modelId = new JsonMapper().readTree(trained).get("model_id").asText();
            click(panel, "预测");
            awaitText(panel, "预测完成");
            assertTrue(text(panel).contains("Iris-setosa"));
            click(panel, "聚类");
            awaitText(panel, "聚类和评估完成");
            assertTrue(text(panel).contains("\"num_clusters\" : 3"));
            click(panel, "保存实验");
            awaitText(panel, "实验已保存");
            String experimentId = text(panel, "实验 ID（可选恢复）：");
            assertFalse(experimentId.isBlank());
            click(panel, "导出模型");
            awaitText(panel, "模型已导出：");
            Path exported = Path.of("target/test-workspace/data/exports/" + modelId + ".model");
            assertTrue(Files.isRegularFile(exported));
            BufferedImage screenshot = new BufferedImage(frame.getWidth(), frame.getHeight(), BufferedImage.TYPE_INT_RGB);
            SwingUtilities.invokeAndWait(() -> frame.paint(screenshot.getGraphics()));
            javax.imageio.ImageIO.write(screenshot, "png", Path.of("target/gui-workflow.png").toFile());
            click(panel, "停止");
            awaitText(panel, "MCP 已停止");
            click(panel, "启动 MCP");
            awaitText(panel, "MCP 已就绪");
            click(panel, "恢复模型");
            awaitText(panel, "模型已恢复");
            assertEquals(experimentId, text(panel, "实验 ID（可选恢复）："));
        } finally {
            SwingUtilities.invokeAndWait(frame::dispose);
        }
        await(() -> serviceProcesses() == 0, 30, "MCP service did not exit with Explorer");
    }

    private static void click(Container panel, String title) throws Exception {
        AbstractButton button = find(panel, AbstractButton.class, title);
        assertNotNull(button, "Missing button " + title);
        SwingUtilities.invokeAndWait(button::doClick);
    }

    private static JTabbedPane tabGroup(Container root) {
        JTabbedPane result = find(root, JTabbedPane.class);
        assertNotNull(result);
        return result;
    }

    private static String text(Container panel) throws Exception {
        JTextArea area = find(panel, JTextArea.class);
        return onEdt(area::getText);
    }

    private static String text(Container panel, String label) throws Exception {
        return onEdt(() -> ((javax.swing.JTextField) labelled(panel, label)).getText().trim());
    }

    private static Component labelled(Container root, String label) {
        for (Component component : root.getComponents()) {
            if (component instanceof JLabel candidate && label.equals(candidate.getText())) {
                Container parent = candidate.getParent();
                int index = List.of(parent.getComponents()).indexOf(candidate);
                if (index + 1 < parent.getComponentCount()) {
                    return parent.getComponent(index + 1);
                }
            }
            if (component instanceof Container container) {
                Component result = labelled(container, label);
                if (result != null) {
                    return result;
                }
            }
        }
        return null;
    }

    private void awaitText(Container panel, String expected) throws Exception {
        assertNotNull(status);
        try {
            await(() -> onEdtUnchecked(() -> {
                if (status.getText().equals("操作失败")) {
                    throw new AssertionError(onEdtUnchecked(() -> find(panel, JTextArea.class).getText()));
                }
                return expected.endsWith("：") ? status.getText().startsWith(expected)
                        : status.getText().equals(expected);
            }), 45, "Timed out");
        } catch (AssertionError error) {
            throw new AssertionError("Expected status " + expected + ", actual " + onEdt(status::getText)
                    + ", output " + text(panel) + ", service log " + serviceLog(panel), error);
        }
    }

    private static void await(BooleanSupplier condition, int seconds, String message) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(50);
        }
        throw new AssertionError(message);
    }

    private static String serviceLog(Container panel) {
        try {
            var field = McpPanel.class.getDeclaredField("controller");
            field.setAccessible(true);
            var controller = (McpPanelController) field.get(panel);
            var threads = new StringBuilder();
            Thread.getAllStackTraces().forEach((thread, stack) -> {
                if (thread.getName().startsWith("weka-mcp-gui") || thread.getName().startsWith("mcp-gui")) {
                    threads.append(thread.getName()).append(' ').append(thread.getState());
                    for (StackTraceElement frame : stack) {
                        threads.append("\n  ").append(frame);
                    }
                }
            });
            var dump = new StringBuilder();
            for (ProcessHandle child : ProcessHandle.allProcesses()
                    .filter(process -> process.info().commandLine().orElse("")
                            .contains("weka-mcp-java-1.6.0.jar"))
                    .filter(process -> process.info().commandLine().orElse("")
                            .contains("test-weka-home/packages/wekaMcp")).toList()) {
                Process jcmd = new ProcessBuilder("jcmd", Long.toString(child.pid()), "Thread.print")
                        .redirectErrorStream(true).start();
                dump.append(new String(jcmd.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
                jcmd.waitFor(10, TimeUnit.SECONDS);
            }
            Files.writeString(Path.of("target/service-thread-dump.txt"), dump);
            return "busy=" + controller.isBusy() + ", running=" + controller.isRunning()
                    + "\n" + threads + "\n" + controller.stderr();
        } catch (Exception error) {
            return error.toString();
        }
    }

    private static long serviceProcesses() {
        return ProcessHandle.allProcesses()
                .filter(process -> process.info().commandLine().orElse("")
                        .contains("weka-mcp-java-1.6.0.jar"))
                .filter(process -> process.info().commandLine().orElse("")
                        .contains("test-weka-home/packages/wekaMcp"))
                .count();
    }

    private static <T> T onEdt(Supplier<T> action) throws Exception {
        if (SwingUtilities.isEventDispatchThread()) {
            return action.get();
        }
        var result = new java.util.concurrent.atomic.AtomicReference<T>();
        SwingUtilities.invokeAndWait(() -> result.set(action.get()));
        return result.get();
    }

    private static <T> T onEdtUnchecked(Supplier<T> action) {
        try {
            return onEdt(action);
        } catch (Exception error) {
            throw new RuntimeException(error);
        }
    }

    private static <T extends Component> T find(Container root, Class<T> type) {
        return find(root, type, null);
    }

    private static <T extends Component> T find(Container root, Class<T> type, String title) {
        List<T> found = new ArrayList<>();
        findInto(root, type, title, found);
        return found.isEmpty() ? null : found.get(0);
    }

    private static <T extends Component> void findInto(Container root, Class<T> type, String title, List<T> found) {
        for (Component component : root.getComponents()) {
            if (type.isInstance(component)) {
                if (title == null || title.equals(component instanceof AbstractButton button
                        ? button.getText() : component instanceof JLabel label ? label.getText() : null)) {
                    found.add(type.cast(component));
                }
            }
            if (component instanceof Container container) {
                findInto(container, type, title, found);
            }
        }
    }
}
