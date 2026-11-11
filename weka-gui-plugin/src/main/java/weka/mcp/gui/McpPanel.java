package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.Logger;
import weka.gui.explorer.Explorer;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class McpPanel extends JPanel implements Explorer.ExplorerPanel, Explorer.LogHandler {
    private static final JsonMapper JSON = new JsonMapper();
    private final PluginWorkspace workspace;
    private final McpPanelController controller;
    private final JButton bridgeButton = new JButton("启动 Agent 桥接");
    private ExplorerBridgeServer bridge;
    private final JTextField datasetField = new JTextField(28);
    private final JTextField classIndexField = new JTextField("0", 4);
    private final JTextField classifierField = new JTextField("weka.classifiers.trees.J48", 28);
    private final JTextField optionsField = new JTextField("", 18);
    private final JTextField experimentField = new JTextField(20);
    private final JComboBox<String> exportFormat = new JComboBox<>(new String[]{"weka_bundle", "raw_model", "pmml"});
    private final JButton startButton = new JButton("启动 MCP");
    private final JButton stopButton = new JButton("停止");
    private final JButton chooseButton = new JButton("选择 ARFF/CSV…");
    private final JButton loadButton = new JButton("加载与拆分");
    private final JButton trainButton = new JButton("训练与评估");
    private final JButton predictButton = new JButton("预测");
    private final JButton clusterButton = new JButton("聚类");
    private final JButton saveButton = new JButton("保存实验");
    private final JButton restoreButton = new JButton("恢复模型");
    private final JButton exportButton = new JButton("导出模型");
    private final JLabel statusLabel = new JLabel("MCP 未启动");
    private final JProgressBar progress = new JProgressBar();
    private final JTextArea output = new JTextArea(16, 72);
    private final List<JButton> operationButtons = new ArrayList<>();
    private Explorer explorer;
    private Logger log;
    private Instances instances;
    private volatile String datasetId;
    private volatile String testDatasetId;
    private volatile String modelId;
    private volatile String clustererId;
    private boolean ready;

    public McpPanel() throws Exception {
        this(PluginWorkspace.installed());
    }

    public McpPanel(PluginWorkspace workspace) {
        this.workspace = workspace;
        this.controller = new McpPanelController(workspace);
        controller.setLogger(message -> {
            if (log != null) {
                log.logMessage(message);
            }
        });
        setLayout(new BorderLayout(6, 6));
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        output.setEditable(false);
        output.setLineWrap(true);
        output.setWrapStyleWord(true);
        progress.setIndeterminate(false);
        operationButtons.addAll(List.of(loadButton, trainButton, predictButton, clusterButton, saveButton,
                restoreButton, exportButton, chooseButton));
        add(buildControls(), BorderLayout.NORTH);
        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("结果", new JScrollPane(output));
        JPanel logPanel = new JPanel(new GridLayout(0, 1));
        logPanel.add(new JLabel("工作目录：" + workspace.workspaceDirectory()));
        logPanel.add(new JLabel("MCP：" + workspace.serviceJar()));
        logPanel.add(new JLabel("Agent 桥接（默认关闭）：" + System.getProperty("weka.mcp.gui.bridge.file",
                workspace.workspaceDirectory().resolve("bridge.json").toString())));
        tabs.addTab("运行环境", logPanel);
        add(tabs, BorderLayout.CENTER);
        JPanel bottom = new JPanel(new BorderLayout());
        bottom.add(statusLabel, BorderLayout.WEST);
        bottom.add(progress, BorderLayout.EAST);
        add(bottom, BorderLayout.SOUTH);
        startButton.addActionListener(event -> startService());
        stopButton.addActionListener(event -> stopService());
        bridgeButton.addActionListener(event -> toggleBridge());
        chooseButton.addActionListener(event -> chooseDataset());
        loadButton.addActionListener(event -> loadDataset());
        trainButton.addActionListener(event -> trainModel());
        predictButton.addActionListener(event -> predict());
        clusterButton.addActionListener(event -> cluster());
        saveButton.addActionListener(event -> saveExperiment());
        restoreButton.addActionListener(event -> restoreExperiment());
        exportButton.addActionListener(event -> exportModel());
        updateControls();
    }

    private JPanel buildControls() {
        JPanel panel = new JPanel(new GridLayout(0, 1, 4, 4));
        JPanel service = new JPanel(new FlowLayout(FlowLayout.LEFT));
        service.add(startButton);
        service.add(stopButton);
        service.add(bridgeButton);
        service.add(new JLabel("实验 ID（可选恢复）："));
        service.add(experimentField);
        service.add(restoreButton);
        panel.add(service);

        JPanel dataset = new JPanel(new FlowLayout(FlowLayout.LEFT));
        dataset.add(chooseButton);
        dataset.add(datasetField);
        dataset.add(new JLabel("类别索引（0=最后）"));
        dataset.add(classIndexField);
        dataset.add(loadButton);
        panel.add(dataset);

        JPanel training = new JPanel(new FlowLayout(FlowLayout.LEFT));
        training.add(new JLabel("分类器"));
        training.add(classifierField);
        training.add(new JLabel("选项"));
        training.add(optionsField);
        training.add(trainButton);
        training.add(predictButton);
        training.add(clusterButton);
        panel.add(training);

        JPanel models = new JPanel(new FlowLayout(FlowLayout.LEFT));
        models.add(saveButton);
        models.add(new JLabel("导出格式"));
        models.add(exportFormat);
        models.add(exportButton);
        panel.add(models);
        return panel;
    }

    private void toggleBridge() {
        try {
            if (bridge != null && bridge.isRunning()) {
                bridge.close();
            } else {
                bridge = new ExplorerBridgeServer(workspace, controller, this::getExplorer);
                bridge.start();
            }
            bridgeButton.setText(bridge.isRunning() ? "停止 Agent 桥接" : "启动 Agent 桥接");
            if (log != null) {
                log.statusMessage(bridge.isRunning() ? "Weka MCP native bridge enabled"
                        : "Weka MCP native bridge disabled");
            }
        } catch (Exception error) {
            showError(error);
        }
    }

    private void startService() {
        setBusy(true, "正在启动 MCP…");
        controller.start(() -> {
            ready = true;
            setBusy(false, "MCP 已就绪");
        }, this::showError);
    }

    private void stopService() {
        controller.stop(() -> {
            ready = false;
            datasetId = null;
            testDatasetId = null;
            modelId = null;
            clustererId = null;
            setBusy(false, "MCP 已停止");
        }, this::showError);
    }

    private void chooseDataset() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            instances = null;
            datasetField.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    private void loadDataset() {
        setBusy(true, "正在加载、画像和拆分数据…");
        controller.execute(() -> {
            String relative = instances == null
                    ? workspace.importDataset(Path.of(datasetField.getText()))
                    : workspace.writeInstances("explorer-" + java.util.UUID.randomUUID() + ".arff", instances);
            Map<String, Object> arguments = new LinkedHashMap<>();
            arguments.put("path", relative);
            arguments.put("format", relative.toLowerCase().endsWith(".csv") ? "csv" : "arff");
            if (classIndex() > 0) {
                arguments.put("class_index", classIndex());
            }
            JsonNode loaded = controller.callTool("weka_load_dataset", arguments);
            datasetId = loaded.get("dataset_id").asText();
            testDatasetId = datasetId;
            ObjectNode result = (ObjectNode) loaded.deepCopy();
            result.set("profile", controller.callTool("weka_profile_dataset", Map.of("dataset_id", datasetId)));
            JsonNode split = controller.callTool("weka_split_dataset", Map.of(
                    "dataset_id", datasetId, "train_percent", 80, "seed", 1));
            datasetId = split.get("training_dataset_id").asText();
            testDatasetId = split.get("test_dataset_id").asText();
            result.set("split", split);
            return result;
        }, result -> {
            output.setText(pretty(result));
            setBusy(false, "数据已加载，训练/测试拆分完成");
        }, this::showError);
    }

    private void trainModel() {
        if (datasetId == null) {
            showError(new IllegalStateException("请先加载数据"));
            return;
        }
        setBusy(true, "正在训练和评估分类器…");
        controller.execute(() -> {
            Map<String, Object> arguments = new LinkedHashMap<>();
            arguments.put("dataset_id", datasetId);
            arguments.put("classifier_class", classifierField.getText().trim());
            arguments.put("options", options());
            if (classIndex() > 0) {
                arguments.put("class_index", classIndex());
            }
            arguments.put("auto_preprocess", true);
            arguments.put("persist", true);
            JsonNode trained = controller.callTool("weka_train_classifier", arguments);
            modelId = trained.get("model_id").asText();
            ObjectNode result = (ObjectNode) trained.deepCopy();
            result.set("evaluation", controller.callTool("weka_evaluate_model", Map.of(
                    "model_id", modelId, "test_dataset_id", testDatasetId)));
            return result;
        }, result -> {
            if (result.has("experiment_id")) {
                experimentField.setText(result.get("experiment_id").asText());
            }
            output.setText(pretty(result));
            setBusy(false, "分类器训练与独立评估完成");
        }, this::showError);
    }

    private void predict() {
        if (modelId == null) {
            showError(new IllegalStateException("请先训练或恢复模型"));
            return;
        }
        setBusy(true, "正在预测…");
        controller.execute(() -> {
            Map<String, Object> arguments = new LinkedHashMap<>();
            arguments.put("model_id", modelId);
            arguments.put("include_distribution", true);
            if (instances != null && instances.numInstances() > 0) {
                arguments.put("instance_json", instanceJson(instances.instance(0), instances));
            } else {
                arguments.put("dataset_id", testDatasetId);
            }
            return controller.callTool("weka_predict", arguments);
        }, result -> {
            output.setText(pretty(result));
            setBusy(false, "预测完成");
        }, this::showError);
    }

    private void cluster() {
        if (datasetId == null) {
            showError(new IllegalStateException("请先加载数据"));
            return;
        }
        setBusy(true, "正在聚类并计算指标…");
        controller.execute(() -> {
            JsonNode clustered = controller.callTool("weka_cluster", Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3, "remove_class", true, "persist", true));
            clustererId = clustered.get("clusterer_id").asText();
            ObjectNode result = (ObjectNode) clustered.deepCopy();
            result.set("evaluation", controller.callTool("weka_cluster_evaluate", Map.of(
                    "clusterer_id", clustererId, "dataset_id", datasetId, "max_samples", 500)));
            return result;
        }, result -> {
            output.setText(pretty(result));
            setBusy(false, "聚类和评估完成");
        }, this::showError);
    }

    private void saveExperiment() {
        if (modelId == null) {
            showError(new IllegalStateException("请先训练或恢复模型"));
            return;
        }
        setBusy(true, "正在保存实验…");
        controller.callTool("weka_save_model", Map.of("model_id", modelId), result -> {
            experimentField.setText(result.get("experiment_id").asText());
            output.setText(pretty(result));
            setBusy(false, "实验已保存");
        }, this::showError);
    }

    private void restoreExperiment() {
        if (experimentField.getText().isBlank()) {
            showError(new IllegalStateException("请输入要恢复的实验 ID"));
            return;
        }
        setBusy(true, "正在恢复模型…");
        controller.execute(() -> {
            JsonNode restored = controller.callTool("weka_load_experiment_model", Map.of(
                    "experiment_id", experimentField.getText().trim()));
            modelId = restored.get("model_id").asText();
            if (testDatasetId != null) {
                ObjectNode result = (ObjectNode) restored.deepCopy();
                result.set("evaluation", controller.callTool("weka_evaluate_model", Map.of(
                        "model_id", modelId, "test_dataset_id", testDatasetId)));
                return result;
            }
            return restored;
        }, result -> {
            output.setText(pretty(result));
            setBusy(false, "模型已恢复");
        }, this::showError);
    }

    private void exportModel() {
        if (modelId == null) {
            showError(new IllegalStateException("请先训练或恢复模型"));
            return;
        }
        setBusy(true, "正在导出模型…");
        String format = (String) exportFormat.getSelectedItem();
        controller.callTool("weka_export_model", Map.of(
                "model_id", modelId,
                "path", "exports/" + modelId + ("pmml".equals(format) ? ".pmml" : ".model"),
                "format", format), result -> {
            output.setText(pretty(result));
            setBusy(false, "模型已导出：" + result.get("path").asText());
        }, this::showError);
    }

    private int classIndex() {
        return Integer.parseInt(classIndexField.getText().trim());
    }

    private List<String> options() {
        try {
            return Arrays.asList(weka.core.Utils.splitOptions(optionsField.getText().trim()));
        } catch (Exception error) {
            throw new IllegalArgumentException("无法解析分类器选项：" + error.getMessage(), error);
        }
    }

    private static String instanceJson(Instance instance, Instances data) {
        ObjectNode result = JSON.createObjectNode();
        for (int index = 0; index < data.numAttributes(); index++) {
            if (index == data.classIndex()) {
                continue;
            }
            if (instance.isMissing(index)) {
                result.putNull(data.attribute(index).name());
            } else if (data.attribute(index).isNumeric()) {
                result.put(data.attribute(index).name(), instance.value(index));
            } else {
                result.put(data.attribute(index).name(), instance.toString(index));
            }
        }
        return result.toString();
    }

    private static String pretty(JsonNode result) {
        try {
            return JSON.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        } catch (Exception error) {
            return result.toString();
        }
    }

    private void showError(Exception error) {
        setBusy(false, "操作失败");
        output.setText("错误：" + (error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage()));
        if (log != null) {
            log.statusMessage("Weka MCP: " + error.getMessage());
        }
    }

    private void setBusy(boolean busy, String message) {
        statusLabel.setText(message);
        progress.setIndeterminate(busy);
        updateControls(busy);
    }

    private void updateControls() {
        updateControls(controller.isBusy());
    }

    private void updateControls(boolean busy) {
        startButton.setEnabled(!ready && !busy);
        stopButton.setEnabled(ready);
        for (JButton button : operationButtons) {
            button.setEnabled(ready && !busy);
        }
    }

    @Override
    public void setExplorer(Explorer parent) {
        this.explorer = parent;
    }

    @Override
    public Explorer getExplorer() {
        return explorer;
    }

    @Override
    public void setInstances(Instances inst) {
        this.instances = inst;
        if (inst != null) {
            datasetField.setText("Explorer 当前数据：" + inst.numInstances() + " 条");
        }
    }

    @Override
    public String getTabTitle() {
        return "Weka MCP";
    }

    @Override
    public String getTabTitleToolTip() {
        return "Run Weka MCP workflows in an isolated service process";
    }

    @Override
    public void setLog(Logger newLog) {
        this.log = newLog;
    }

    public void close() {
        if (bridge != null) {
            bridge.close();
        }
        controller.close();
    }

    @Override
    public void removeNotify() {
        close();
        super.removeNotify();
    }
}
