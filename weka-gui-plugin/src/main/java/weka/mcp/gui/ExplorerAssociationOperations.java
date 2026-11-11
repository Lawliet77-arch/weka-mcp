package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.associations.Apriori;
import weka.associations.Associator;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.gui.GenericObjectEditor;
import weka.gui.ResultHistoryPanel;
import weka.gui.explorer.AssociationsPanel;
import weka.gui.explorer.Explorer;

import javax.swing.JButton;
import javax.swing.JTextArea;
import java.util.Set;
import java.util.concurrent.TimeUnit;

final class ExplorerAssociationOperations {
    private static final JsonMapper JSON = new JsonMapper();
    private static final Set<String> ASSOCIATORS = Set.of(
            "weka.associations.Apriori", "weka.associations.FPGrowth");
    private final Explorer explorer;
    private String dataScope = "current_dataset";
    private int associationAttributes;
    private ExplorerRunState lastRun;

    ExplorerAssociationOperations(Explorer explorer) {
        this.explorer = explorer;
    }

    void appendState(ObjectNode result) throws Exception {
        AssociationsPanel panel = panel();
        Object configured = ExplorerSession.field(panel, "m_AssociatorEditor", GenericObjectEditor.class).getValue();
        result.put("associator", configured == null ? "" : configured.getClass().getName());
        result.put("associator_running", isRunning());
        result.put("association_result_count", history(panel).getList().getModel().getSize());
        result.put("association_data_scope", dataScope);
        result.put("association_attributes", associationAttributes);
        result.put("task_running", result.path("task_running").booleanValue(false)
                || result.path("associator_running").booleanValue(false));
    }

    ObjectNode train(JsonNode arguments) throws Exception {
        String className = arguments.has("associator_class")
                ? arguments.path("associator_class").asText() : "weka.associations.FPGrowth";
        if (!ASSOCIATORS.contains(className)) {
            throw new IllegalArgumentException("Associator is not enabled for native Explorer automation: "
                    + className + "; enabled=" + ASSOCIATORS);
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
            for (int index = 0; index < data.numAttributes(); index++) {
                if (!data.attribute(index).isNominal()) {
                    throw new IllegalArgumentException("Association algorithms require nominal attributes; "
                            + "apply Discretize before association mining");
                }
            }
            Associator created = (Associator) Class.forName(className).getDeclaredConstructor().newInstance();
            if (created instanceof OptionHandler handler) {
                handler.setOptions(options);
            } else if (options.length > 0) {
                throw new IllegalArgumentException("Associator does not accept options: " + className);
            }
            if (arguments.has("num_rules")) {
                if (!arguments.get("num_rules").isInt()) {
                    throw new IllegalArgumentException("num_rules must be an integer");
                }
                int count = arguments.get("num_rules").intValue();
                if (count < 1 || count > 100) {
                    throw new IllegalArgumentException("num_rules must be between 1 and 100");
                }
                if (created instanceof FPGrowth fpGrowth) {
                    fpGrowth.setNumRulesToFind(count);
                } else if (created instanceof Apriori apriori) {
                    apriori.setNumRules(count);
                }
            }
            AssociationsPanel panel = panel();
            ExplorerSession.field(panel, "m_AssociatorEditor", GenericObjectEditor.class).setValue(created);
            Instances trainingData = data;
            boolean expanded = false;
            String scope = "current_dataset";
            if (created instanceof FPGrowth) {
                ExplorerAssociationData.EncodedData encoded = ExplorerAssociationData.forFpGrowth(data);
                trainingData = encoded.instances();
                expanded = encoded.expanded();
                scope = "panel_only_binary_nominal";
            }
            panel.setInstances(trainingData);
            explorer.getTabbedPane().setSelectedComponent(panel);
            JButton start = ExplorerSession.field(panel, "m_StartBut", JButton.class);
            if (!start.isEnabled()) {
                throw new IllegalStateException("Explorer rejected the association data or configuration; "
                        + "FPGrowth requires unary or binary nominal attributes");
            }
            int historyAtStart = history(panel).getList().getModel().getSize();
            start.doClick();
            lastRun = new ExplorerRunState(java.util.UUID.randomUUID().toString(),
                    ExplorerResultMetadata.snapshot(data), historyAtStart);
            dataScope = scope;
            associationAttributes = trainingData.numAttributes();
            ObjectNode result = JSON.createObjectNode();
            result.put("operation_id", lastRun.id);
            result.put("training_dataset_version", lastRun.dataset.path("dataset_version").asText());
            result.put("operation", "association_mining");
            result.put("status", "RUNNING");
            result.put("associator_class", className);
            result.put("binary_nominal_expansion", expanded);
            result.put("data_scope", scope);
            result.put("association_attributes", trainingData.numAttributes());
            result.put("associator_running", isRunning());
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
            AssociationsPanel panel = panel();
            ResultHistoryPanel history = history(panel);
            int count = history.getList().getModel().getSize();
            boolean running = isRunning();
            boolean complete = !running && (lastRun == null ? count > 0
                    : !lastRun.cancelled && count > lastRun.historyAtStart);
            ObjectNode result = JSON.createObjectNode();
            result.put("associator_running", running);
            result.put("result_count", count);
            result.put("complete", complete);
            result.put("partial", running);
            result.put("data_scope", dataScope);
            result.put("association_attributes", associationAttributes);
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
            AssociationsPanel panel = panel();
            boolean running = isRunning();
            if (running) {
                if (lastRun != null) {
                    lastRun.cancelled = true;
                }
                ExplorerSession.field(panel, "m_StopBut", JButton.class).doClick();
            }
            ObjectNode result = JSON.createObjectNode();
            result.put("stop_requested", running);
            result.put("associator_running", isRunning());
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

    private AssociationsPanel panel() {
        return explorer.getPanels().stream().filter(AssociationsPanel.class::isInstance)
                .map(AssociationsPanel.class::cast).findFirst()
                .orElseThrow(() -> new IllegalStateException("This Weka version has no Associate panel"));
    }

    private static ResultHistoryPanel history(AssociationsPanel panel) throws Exception {
        return ExplorerSession.field(panel, "m_History", ResultHistoryPanel.class);
    }

    static ObjectNode trainProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode associator = properties.putObject("associator_class");
        associator.put("type", "string");
        associator.put("description", "Apriori or FPGrowth; defaults to FPGrowth");
        ObjectNode rules = properties.putObject("num_rules");
        rules.put("type", "integer");
        rules.put("description", "Maximum number of rules, 1 to 100; uses the Weka default when omitted");
        ObjectNode options = properties.putObject("options");
        options.put("type", "array");
        ObjectNode items = options.putObject("items");
        items.put("type", "string");
        items.put("description", "Weka associator option");
        return properties;
    }

    static ObjectNode resultProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode wait = properties.putObject("wait_seconds");
        wait.put("type", "integer");
        wait.put("description", "Seconds to wait for the native association task, 0 to 150");
        return properties;
    }
}
