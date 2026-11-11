package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.gui.GenericObjectEditor;
import weka.gui.ResultHistoryPanel;
import weka.gui.explorer.AttributeSelectionPanel;
import weka.gui.explorer.Explorer;

import javax.swing.JButton;
import javax.swing.JRadioButton;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import java.util.Map;
import java.util.concurrent.TimeUnit;

final class ExplorerAttributeSelectionOperations {
    private static final JsonMapper JSON = new JsonMapper();
    private static final Map<String, String> EVALUATORS = Map.of(
            "weka.attributeSelection.CfsSubsetEval", "weka.attributeSelection.BestFirst",
            "weka.attributeSelection.GainRatioAttributeEval", "weka.attributeSelection.Ranker",
            "weka.attributeSelection.InfoGainAttributeEval", "weka.attributeSelection.Ranker");
    private final Explorer explorer;
    private String evaluatorClass = "";
    private String searchClass = "";
    private int folds;
    private int numToSelect;
    private ExplorerRunState lastRun;

    ExplorerAttributeSelectionOperations(Explorer explorer) {
        this.explorer = explorer;
    }

    void appendState(ObjectNode result) throws Exception {
        AttributeSelectionPanel panel = panel();
        Object evaluator = ExplorerSession.field(panel, "m_AttributeEvaluatorEditor", GenericObjectEditor.class).getValue();
        Object search = ExplorerSession.field(panel, "m_AttributeSearchEditor", GenericObjectEditor.class).getValue();
        result.put("attribute_evaluator", evaluator == null ? "" : evaluator.getClass().getName());
        result.put("attribute_search", search == null ? "" : search.getClass().getName());
        result.put("attribute_selection_running", isRunning());
        result.put("attribute_selection_result_count", history(panel).getList().getModel().getSize());
        result.put("task_running", result.path("task_running").booleanValue(false)
                || result.path("attribute_selection_running").booleanValue(false));
    }

    ObjectNode select(JsonNode arguments) throws Exception {
        String evaluatorName = arguments.has("evaluator_class")
                ? arguments.path("evaluator_class").asText() : "weka.attributeSelection.CfsSubsetEval";
        if (!EVALUATORS.containsKey(evaluatorName)) {
            throw new IllegalArgumentException("Attribute evaluator is not enabled: " + evaluatorName
                    + "; enabled=" + EVALUATORS.keySet());
        }
        String expectedSearch = EVALUATORS.get(evaluatorName);
        String searchName = arguments.has("search_class")
                ? arguments.path("search_class").asText() : expectedSearch;
        if (!expectedSearch.equals(searchName)) {
            throw new IllegalArgumentException("search_class must be " + expectedSearch + " for " + evaluatorName);
        }
        String[] evaluatorOptions = ExplorerSession.options(arguments.get("evaluator_options"));
        String[] searchOptions = ExplorerSession.options(arguments.get("search_options"));
        return ExplorerSession.onEdt(() -> {
            Instances data = explorer.getPreprocessPanel().getInstances();
            if (data == null || data.numInstances() == 0) {
                throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
            }
            if (data.classIndex() < 0) {
                throw new IllegalStateException("Set a class attribute before attribute selection");
            }
            if (new ExplorerSession(explorer).state().path("task_running").booleanValue()) {
                throw new IllegalStateException("Another native Explorer task is already running");
            }
            ASEvaluation evaluator = (ASEvaluation) Class.forName(evaluatorName).getDeclaredConstructor().newInstance();
            ASSearch search = (ASSearch) Class.forName(searchName).getDeclaredConstructor().newInstance();
            evaluator.setOptions(evaluatorOptions);
            search.setOptions(searchOptions);
            int requestedFolds = arguments.has("folds") ? arguments.path("folds").intValue() : 0;
            if (arguments.has("folds") && !arguments.get("folds").isInt()) {
                throw new IllegalArgumentException("folds must be an integer");
            }
            if (requestedFolds != 0 && (requestedFolds < 2 || requestedFolds > 50)) {
                throw new IllegalArgumentException("folds must be between 2 and 50, or omitted for full evaluation");
            }
            if (arguments.has("seed") && requestedFolds == 0) {
                throw new IllegalArgumentException("seed is only used with cross-validation folds");
            }
            if (arguments.has("seed") && !arguments.get("seed").isInt()) {
                throw new IllegalArgumentException("seed must be an integer");
            }
            int seed = arguments.has("seed") ? arguments.get("seed").intValue() : 1;
            int selected = 0;
            if (arguments.has("num_to_select")) {
                if (!(search instanceof Ranker)) {
                    throw new IllegalArgumentException("num_to_select is only supported for Ranker");
                }
                if (!arguments.get("num_to_select").isInt()) {
                    throw new IllegalArgumentException("num_to_select must be an integer");
                }
                selected = arguments.get("num_to_select").intValue();
                if (selected < 1 || selected >= data.numAttributes()) {
                    throw new IllegalArgumentException("num_to_select must be between 1 and "
                            + (data.numAttributes() - 1));
                }
                ((Ranker) search).setNumToSelect(selected);
            }
            AttributeSelectionPanel panel = panel();
            configure(panel, evaluator, search);
            ExplorerSession.field(panel, "m_TrainBut", JRadioButton.class).setSelected(requestedFolds == 0);
            ExplorerSession.field(panel, "m_CVBut", JRadioButton.class).setSelected(requestedFolds != 0);
            if (requestedFolds != 0) {
                ExplorerSession.field(panel, "m_CVText", JTextField.class).setText(Integer.toString(requestedFolds));
                ExplorerSession.field(panel, "m_SeedText", JTextField.class).setText(Integer.toString(seed));
            }
            panel.setInstances(data);
            explorer.getTabbedPane().setSelectedComponent(panel);
            JButton start = ExplorerSession.field(panel, "m_StartBut", JButton.class);
            if (!start.isEnabled()) {
                throw new IllegalStateException("Explorer rejected the attribute selection data or configuration");
            }
            int historyAtStart = history(panel).getList().getModel().getSize();
            start.doClick();
            lastRun = new ExplorerRunState(java.util.UUID.randomUUID().toString(),
                    ExplorerResultMetadata.snapshot(data), historyAtStart);
            evaluatorClass = evaluatorName;
            searchClass = searchName;
            folds = requestedFolds;
            numToSelect = selected;
            ObjectNode result = JSON.createObjectNode();
            result.put("operation_id", lastRun.id);
            result.put("training_dataset_version", lastRun.dataset.path("dataset_version").asText());
            result.put("operation", "attribute_selection");
            result.put("status", "RUNNING");
            result.put("evaluator_class", evaluatorName);
            result.put("search_class", searchName);
            result.put("evaluation_mode", requestedFolds == 0 ? "full_training_set" : "cross_validation");
            if (requestedFolds != 0) {
                result.put("folds", requestedFolds);
                result.put("seed", seed);
            }
            if (selected > 0) {
                result.put("num_to_select", selected);
            }
            result.put("attribute_selection_running", isRunning());
            return result;
        });
    }

    private static void configure(AttributeSelectionPanel panel, ASEvaluation evaluator, ASSearch search)
            throws Exception {
        GenericObjectEditor evaluatorEditor = ExplorerSession.field(panel, "m_AttributeEvaluatorEditor",
                GenericObjectEditor.class);
        GenericObjectEditor searchEditor = ExplorerSession.field(panel, "m_AttributeSearchEditor",
                GenericObjectEditor.class);
        var paused = new java.util.ArrayList<PausedListener>();
        for (GenericObjectEditor editor : java.util.List.of(evaluatorEditor, searchEditor)) {
            for (java.beans.PropertyChangeListener listener : ExplorerSession.field(editor, "m_Support",
                    java.beans.PropertyChangeSupport.class).getPropertyChangeListeners()) {
                if (listener.getClass().getName().startsWith("weka.gui.explorer.AttributeSelectionPanel$")) {
                    editor.removePropertyChangeListener(listener);
                    paused.add(new PausedListener(editor, listener));
                }
            }
        }
        try {
            searchEditor.setValue(search);
            evaluatorEditor.setValue(evaluator);
        } finally {
            for (PausedListener listener : paused) {
                listener.editor().addPropertyChangeListener(listener.listener());
            }
        }
    }

    private record PausedListener(GenericObjectEditor editor, java.beans.PropertyChangeListener listener) {
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
            AttributeSelectionPanel panel = panel();
            ResultHistoryPanel history = history(panel);
            int count = history.getList().getModel().getSize();
            boolean running = isRunning();
            boolean complete = !running && (lastRun == null ? count > 0
                    : !lastRun.cancelled && count > lastRun.historyAtStart);
            ObjectNode result = JSON.createObjectNode();
            result.put("attribute_selection_running", running);
            result.put("result_count", count);
            result.put("complete", complete);
            result.put("partial", running);
            result.put("evaluator_class", evaluatorClass);
            result.put("search_class", searchClass);
            result.put("evaluation_mode", folds == 0 ? "full_training_set" : "cross_validation");
            if (folds != 0) {
                result.put("folds", folds);
            }
            if (numToSelect > 0) {
                result.put("num_to_select", numToSelect);
            }
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
            AttributeSelectionPanel panel = panel();
            boolean running = isRunning();
            if (running) {
                if (lastRun != null) {
                    lastRun.cancelled = true;
                }
                ExplorerSession.field(panel, "m_StopBut", JButton.class).doClick();
            }
            ObjectNode result = JSON.createObjectNode();
            result.put("stop_requested", running);
            result.put("attribute_selection_running", isRunning());
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

    private AttributeSelectionPanel panel() {
        return explorer.getPanels().stream().filter(AttributeSelectionPanel.class::isInstance)
                .map(AttributeSelectionPanel.class::cast).findFirst()
                .orElseThrow(() -> new IllegalStateException("This Weka version has no Select attributes panel"));
    }

    private static ResultHistoryPanel history(AttributeSelectionPanel panel) throws Exception {
        return ExplorerSession.field(panel, "m_History", ResultHistoryPanel.class);
    }

    static ObjectNode selectionProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode evaluator = properties.putObject("evaluator_class");
        evaluator.put("type", "string");
        evaluator.put("description", "CfsSubsetEval, GainRatioAttributeEval or InfoGainAttributeEval");
        ObjectNode search = properties.putObject("search_class");
        search.put("type", "string");
        search.put("description", "BestFirst for CfsSubsetEval, Ranker for the other evaluators");
        ObjectNode folds = properties.putObject("folds");
        folds.put("type", "integer");
        folds.put("description", "Cross-validation folds, 2 to 50; omit for full training-set evaluation");
        ObjectNode seed = properties.putObject("seed");
        seed.put("type", "integer");
        seed.put("description", "Cross-validation seed; defaults to 1");
        ObjectNode selected = properties.putObject("num_to_select");
        selected.put("type", "integer");
        selected.put("description", "Number of attributes returned by Ranker");
        ObjectNode evaluatorOptions = properties.putObject("evaluator_options");
        evaluatorOptions.put("type", "array");
        evaluatorOptions.withArray("items").addObject().put("type", "string");
        ObjectNode searchOptions = properties.putObject("search_options");
        searchOptions.put("type", "array");
        searchOptions.withArray("items").addObject().put("type", "string");
        return properties;
    }

    static ObjectNode resultProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode wait = properties.putObject("wait_seconds");
        wait.put("type", "integer");
        wait.put("description", "Seconds to wait for the attribute selection task, 0 to 150");
        return properties;
    }
}
