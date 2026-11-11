package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.clusterers.Clusterer;
import weka.clusterers.RandomizableClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.gui.GenericObjectEditor;
import weka.gui.ResultHistoryPanel;
import weka.gui.explorer.ClustererPanel;
import weka.gui.explorer.Explorer;

import javax.swing.JButton;
import javax.swing.JTextArea;
import java.util.Set;
import java.util.concurrent.TimeUnit;

final class ExplorerClusterOperations {
    private static final JsonMapper JSON = new JsonMapper();
    private static final Set<String> CLUSTERERS = Set.of(
            "weka.clusterers.SimpleKMeans",
            "weka.clusterers.FarthestFirst",
            "weka.clusterers.Canopy",
            "weka.clusterers.Cobweb");
    private final Explorer explorer;
    private ExplorerRunState lastRun;

    ExplorerClusterOperations(Explorer explorer) {
        this.explorer = explorer;
    }

    void appendState(ObjectNode result) throws Exception {
        ClustererPanel panel = panel();
        Object configured = ExplorerSession.field(panel, "m_ClustererEditor", GenericObjectEditor.class).getValue();
        result.put("clusterer", configured == null ? "" : configured.getClass().getName());
        result.put("clusterer_running", isRunning());
        result.put("cluster_result_count", history(panel).getList().getModel().getSize());
        Thread io = ExplorerSession.field(explorer.getPreprocessPanel(), "m_IOThread", Thread.class);
        result.put("preprocess_running", io != null && io.isAlive());
        result.put("task_running", result.path("classifier_running").booleanValue(false)
                || result.path("clusterer_running").booleanValue(false)
                || result.path("preprocess_running").booleanValue(false));
    }

    ObjectNode train(JsonNode arguments) throws Exception {
        String className = arguments.has("clusterer_class")
                ? arguments.path("clusterer_class").asText() : "weka.clusterers.SimpleKMeans";
        if (!CLUSTERERS.contains(className)) {
            throw new IllegalArgumentException("Clusterer is not enabled for native Explorer automation: "
                    + className + "; enabled=" + CLUSTERERS);
        }
        String[] options = ExplorerSession.options(arguments.get("options"));
        return ExplorerSession.onEdt(() -> {
            Instances data = explorer.getPreprocessPanel().getInstances();
            if (data == null || data.numInstances() == 0) {
                throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
            }
            if (new ExplorerSession(explorer).state().path("task_running").booleanValue()) {
                throw new IllegalStateException("Another native Explorer task is already running");
            }
            Clusterer created = (Clusterer) Class.forName(className).getDeclaredConstructor().newInstance();
            if (created instanceof OptionHandler handler) {
                handler.setOptions(options);
            } else if (options.length > 0) {
                throw new IllegalArgumentException("Clusterer does not accept options: " + className);
            }
            if (created instanceof RandomizableClusterer randomizable) {
                if (arguments.has("seed")) {
                    if (!arguments.get("seed").isInt()) {
                        throw new IllegalArgumentException("seed must be an integer");
                    }
                    randomizable.setSeed(arguments.get("seed").intValue());
                } else {
                    randomizable.setSeed(1);
                }
            } else if (arguments.has("seed")) {
                throw new IllegalArgumentException("This clusterer does not support a random seed");
            }
            if (created instanceof SimpleKMeans kmeans) {
                int clusters = arguments.has("num_clusters") ? arguments.path("num_clusters").intValue() : 3;
                if (arguments.has("num_clusters") && !arguments.get("num_clusters").isInt()) {
                    throw new IllegalArgumentException("num_clusters must be an integer");
                }
                if (clusters < 1 || clusters > Math.min(100, data.numInstances())) {
                    throw new IllegalArgumentException("num_clusters must be between 1 and "
                            + Math.min(100, data.numInstances()));
                }
                kmeans.setNumClusters(clusters);
            } else if (arguments.has("num_clusters")) {
                throw new IllegalArgumentException("num_clusters is only supported for SimpleKMeans");
            }
            ClustererPanel panel = panel();
            ExplorerSession.field(panel, "m_ClustererEditor", GenericObjectEditor.class).setValue(created);
            explorer.getTabbedPane().setSelectedComponent(panel);
            int historyAtStart = history(panel).getList().getModel().getSize();
            ExplorerSession.field(panel, "m_StartBut", JButton.class).doClick();
            lastRun = new ExplorerRunState(java.util.UUID.randomUUID().toString(),
                    ExplorerResultMetadata.snapshot(data), historyAtStart);
            ObjectNode result = JSON.createObjectNode();
            result.put("operation", "clusterer_training");
            result.put("operation_id", lastRun.id);
            result.put("training_dataset_version", lastRun.dataset.path("dataset_version").asText());
            result.put("status", "RUNNING");
            result.put("clusterer_class", className);
            if (created instanceof SimpleKMeans kmeans) {
                result.put("num_clusters", kmeans.getNumClusters());
            }
            result.put("clusterer_running", isRunning());
            return result;
        });
    }

    ObjectNode results(JsonNode arguments) throws Exception {
        int waitSeconds = arguments.has("wait_seconds") ? arguments.path("wait_seconds").intValue() : 0;
        if (arguments.has("wait_seconds") && !arguments.get("wait_seconds").isInt()) {
            throw new IllegalArgumentException("wait_seconds must be an integer");
        }
        if (waitSeconds < 0 || waitSeconds > 150) {
            throw new IllegalArgumentException("wait_seconds must be between 0 and 150");
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(waitSeconds);
        while (ExplorerSession.onEdt(this::isRunning) && System.nanoTime() < deadline) {
            Thread.sleep(50);
        }
        return ExplorerSession.onEdt(() -> {
            ClustererPanel panel = panel();
            ResultHistoryPanel history = history(panel);
            int count = history.getList().getModel().getSize();
            boolean running = isRunning();
            boolean complete = !running && (lastRun == null ? count > 0
                    : !lastRun.cancelled && count > lastRun.historyAtStart);
            ObjectNode result = JSON.createObjectNode();
            result.put("clusterer_running", running);
            result.put("result_count", count);
            result.put("complete", complete);
            result.put("partial", running);
            if (running && lastRun != null) {
                result.put("name", "");
                result.put("output", ExplorerSession.field(panel, "m_OutText", JTextArea.class).getText());
                result.put("model_available", false);
            } else if (count > 0) {
                String name = history.getNameAtIndex(count - 1);
                result.put("name", name);
                StringBuffer buffer = history.getNamedBuffer(name);
                result.put("output", buffer == null ? "" : buffer.toString());
                Object model = history.getNamedObject(name);
                result.put("model_available", model != null);
                if (model != null) {
                    result.put("model_type", model.getClass().getName());
                }
            } else {
                result.put("name", "");
                result.put("output", ExplorerSession.field(panel, "m_OutText", JTextArea.class).getText());
                result.put("model_available", false);
            }
            result.set("evaluation_metrics", ExplorerResultMetadata.evaluationMetrics(result.path("output").asText()));
            ExplorerResultMetadata.addProvenance(result, lastRun == null ? "" : lastRun.id,
                    lastRun == null ? null : lastRun.dataset, running, complete,
                    explorer.getPreprocessPanel().getInstances());
            if (lastRun != null && lastRun.cancelled && !running
                    && !result.path("dataset_changed_since_training").booleanValue()) {
                result.put("result_status", "CANCELLED");
            }
            return result;
        });
    }

    ObjectNode stop() throws Exception {
        return ExplorerSession.onEdt(() -> {
            ClustererPanel panel = panel();
            boolean running = isRunning();
            if (running) {
                if (lastRun != null) {
                    lastRun.cancelled = true;
                }
                ExplorerSession.field(panel, "m_StopBut", JButton.class).doClick();
            }
            ObjectNode result = JSON.createObjectNode();
            result.put("stop_requested", running);
            result.put("clusterer_running", isRunning());
            return result;
        });
    }

    boolean isRunning() throws Exception {
        Thread thread = ExplorerSession.field(panel(), "m_RunThread", Thread.class);
        return thread != null && thread.isAlive();
    }

    void close() {
        try {
            stop();
        } catch (Exception ignored) {
        }
    }

    private ClustererPanel panel() {
        return explorer.getPanels().stream().filter(ClustererPanel.class::isInstance)
                .map(ClustererPanel.class::cast).findFirst()
                .orElseThrow(() -> new IllegalStateException("This Weka version has no Cluster panel"));
    }

    private static ResultHistoryPanel history(ClustererPanel panel) throws Exception {
        return ExplorerSession.field(panel, "m_History", ResultHistoryPanel.class);
    }

    static ObjectNode trainProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode clusterer = properties.putObject("clusterer_class");
        clusterer.put("type", "string");
        clusterer.put("description", "SimpleKMeans, FarthestFirst, Canopy or Cobweb; defaults to SimpleKMeans");
        ObjectNode clusters = properties.putObject("num_clusters");
        clusters.put("type", "integer");
        clusters.put("description", "Number of clusters for SimpleKMeans, 1 to 100 and no more than the instance count");
        ObjectNode seed = properties.putObject("seed");
        seed.put("type", "integer");
        seed.put("description", "Random seed for clusterers that support it; defaults to 1");
        ObjectNode options = properties.putObject("options");
        options.put("type", "array");
        ObjectNode items = options.putObject("items");
        items.put("type", "string");
        items.put("description", "Weka clusterer option");
        return properties;
    }

    static ObjectNode resultProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode wait = properties.putObject("wait_seconds");
        wait.put("type", "integer");
        wait.put("description", "Seconds to wait for the native cluster task, 0 to 150");
        return properties;
    }
}
