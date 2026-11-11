package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.gui.GenericObjectEditor;
import weka.gui.ResultHistoryPanel;
import weka.gui.explorer.ClassifierPanel;
import weka.gui.explorer.Explorer;

import javax.swing.AbstractButton;
import javax.swing.JButton;
import javax.swing.JRadioButton;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.lang.reflect.Field;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class ExplorerSession {
    public static final String STATE = "weka_explorer_get_state";
    public static final String SET_CLASS = "weka_explorer_set_class_index";
    public static final String TRAIN = "weka_explorer_train_classifier";
    public static final String RESULTS = "weka_explorer_get_results";
    public static final String STOP = "weka_explorer_stop_classifier";
    public static final String LOAD = "weka_explorer_load_dataset";
    public static final String FILTER = "weka_explorer_apply_filter";
    public static final String DESCRIBE = "weka_explorer_describe_dataset";
    public static final String UNDO = "weka_explorer_undo";
    public static final String CLUSTER = "weka_explorer_train_clusterer";
    public static final String CLUSTER_RESULTS = "weka_explorer_get_cluster_results";
    public static final String CLUSTER_STOP = "weka_explorer_stop_clusterer";
    public static final String ASSOCIATE = "weka_explorer_run_association";
    public static final String ASSOCIATE_RESULTS = "weka_explorer_get_association_results";
    public static final String ASSOCIATE_STOP = "weka_explorer_stop_association";
    public static final String SELECT_ATTRIBUTES = "weka_explorer_select_attributes";
    public static final String SELECT_RESULTS = "weka_explorer_get_attribute_selection_results";
    public static final String SELECT_STOP = "weka_explorer_stop_attribute_selection";
    public static final String VISUALIZE = "weka_explorer_visualize";
    public static final String CAPTURE_VISUALIZATION = "weka_explorer_capture_visualization";
    public static final String CLOSE_VISUALIZATION = "weka_explorer_close_visualization";
    private static final JsonMapper JSON = new JsonMapper();
    private static final Set<String> CLASSIFIERS = Set.of(
            "weka.classifiers.trees.J48",
            "weka.classifiers.trees.RandomForest",
            "weka.classifiers.bayes.NaiveBayes",
            "weka.classifiers.functions.Logistic",
            "weka.classifiers.functions.MultilayerPerceptron",
            "weka.classifiers.rules.ZeroR");
    private final Explorer explorer;
    private final AtomicBoolean writing = new AtomicBoolean();
    private ClassifierRun lastClassifierRun;

    private final ExplorerDataOperations dataOperations;
    private final ExplorerClusterOperations clusterOperations;
    private final ExplorerAssociationOperations associationOperations;
    private final ExplorerAttributeSelectionOperations attributeOperations;
    private final ExplorerVisualizationOperations visualizationOperations;

    public ExplorerSession(Explorer explorer) {
        this(explorer, null);
    }

    public ExplorerSession(Explorer explorer, PluginWorkspace workspace) {
        if (explorer == null) {
            throw new IllegalArgumentException("The Explorer window is not attached");
        }
        this.explorer = explorer;
        this.dataOperations = new ExplorerDataOperations(explorer, workspace);
        this.clusterOperations = new ExplorerClusterOperations(explorer);
        this.associationOperations = new ExplorerAssociationOperations(explorer);
        this.attributeOperations = new ExplorerAttributeSelectionOperations(explorer);
        this.visualizationOperations = new ExplorerVisualizationOperations(explorer, workspace);
    }

    public ObjectNode call(String tool, JsonNode arguments) throws Exception {
        JsonNode parameters = arguments == null || arguments.isNull() ? JSON.createObjectNode() : arguments;
        if (!parameters.isObject()) {
            throw new IllegalArgumentException("arguments must be a JSON object");
        }
        return switch (tool) {
            case LOAD -> dataOperations.load(parameters);
            case FILTER -> dataOperations.filter(parameters);
            case DESCRIBE -> dataOperations.describe(parameters);
            case UNDO -> dataOperations.undo();
            case STATE -> state();
            case SET_CLASS -> setClassIndex(parameters);
            case TRAIN -> train(parameters);
            case RESULTS -> results(parameters);
            case STOP -> stop();
            case CLUSTER -> clusterOperations.train(parameters);
            case CLUSTER_RESULTS -> clusterOperations.results(parameters);
            case CLUSTER_STOP -> clusterOperations.stop();
            case ASSOCIATE -> associationOperations.train(parameters);
            case ASSOCIATE_RESULTS -> associationOperations.results(parameters);
            case ASSOCIATE_STOP -> associationOperations.stop();
            case SELECT_ATTRIBUTES -> attributeOperations.select(parameters);
            case SELECT_RESULTS -> attributeOperations.results(parameters);
            case SELECT_STOP -> attributeOperations.stop();
            case VISUALIZE -> visualizationOperations.show(parameters);
            case CLOSE_VISUALIZATION -> visualizationOperations.closeTool();
            default -> throw new IllegalArgumentException("Unknown Explorer tool: " + tool);
        };
    }

    public ObjectNode state() throws Exception {
        return onEdt(() -> {
            ObjectNode result = JSON.createObjectNode();
            Instances data = explorer.getPreprocessPanel().getInstances();
            result.put("dataset_loaded", data != null);
            result.put("instances", data == null ? 0 : data.numInstances());
            result.put("attributes", data == null ? 0 : data.numAttributes());
            result.put("relation_name", data == null ? "" : data.relationName());
            result.put("dataset_version", ExplorerResultMetadata.version(data));
            result.put("class_index", data == null || data.classIndex() < 0 ? 0 : data.classIndex() + 1);
            result.put("class_attribute", data == null || data.classIndex() < 0
                    ? "" : data.classAttribute().name());
            result.put("active_tab", explorer.getTabbedPane().getTitleAt(
                    explorer.getTabbedPane().getSelectedIndex()));
            ClassifierPanel classifier = classifier();
            result.put("classifier", classifier.getClassifier().getClass().getName());
            result.put("classifier_running", running(classifier));
            result.put("result_count", classifier.getResultHistory().getList().getModel().getSize());
            clusterOperations.appendState(result);
            associationOperations.appendState(result);
            attributeOperations.appendState(result);
            visualizationOperations.appendState(result);
            return result;
        });
    }

    public ExplorerVisualizationOperations.Capture captureVisualization(JsonNode arguments) throws Exception {
        return visualizationOperations.capture(arguments);
    }

    private ObjectNode setClassIndex(JsonNode arguments) throws Exception {
        int requested = requiredInt(arguments, "class_index");
        return write(() -> onEdt(() -> {
            Instances current = explorer.getPreprocessPanel().getInstances();
            requireDataset(current);
            if (state().path("task_running").booleanValue()) {
                throw new IllegalStateException("Stop the native Explorer task before changing the class index");
            }
            int index = requested == 0 ? current.numAttributes() - 1 : requested - 1;
            if (index < 0 || index >= current.numAttributes()) {
                throw new IllegalArgumentException("class_index must be 0 (last) or between 1 and "
                        + current.numAttributes());
            }
            Instances updated = new Instances(current);
            updated.setClassIndex(index);
            explorer.getPreprocessPanel().setInstances(updated);
            ObjectNode result = state();
            result.put("updated", true);
            return result;
        }));
    }

    private ObjectNode train(JsonNode arguments) throws Exception {
        String className = optionalText(arguments, "classifier_class", "weka.classifiers.trees.J48");
        if (!CLASSIFIERS.contains(className)) {
            throw new IllegalArgumentException("Classifier is not enabled for native Explorer automation: "
                    + className + "; enabled=" + CLASSIFIERS);
        }
        String[] options = options(arguments.get("options"));
        int folds = optionalInt(arguments, "folds", 10);
        int seed = optionalInt(arguments, "seed", 1);
        if (folds < 2 || folds > 50) {
            throw new IllegalArgumentException("folds must be between 2 and 50");
        }
        return write(() -> onEdt(() -> {
            Instances data = explorer.getPreprocessPanel().getInstances();
            requireDataset(data);
            ClassifierPanel panel = classifier();
            if (state().path("task_running").booleanValue()) {
                throw new IllegalStateException("Another native Explorer task is already running");
            }
            Classifier created = "weka.classifiers.rules.ZeroR".equals(className)
                    ? new ZeroR() : (Classifier) Class.forName(className).getDeclaredConstructor().newInstance();
            if (created instanceof OptionHandler handler) {
                handler.setOptions(options);
            } else if (options.length > 0) {
                throw new IllegalArgumentException("Classifier does not accept options: " + className);
            }
            field(panel, "m_ClassifierEditor", GenericObjectEditor.class).setValue(created);
            field(panel, "m_CVBut", JRadioButton.class).setSelected(true);
            field(panel, "m_CVText", JTextField.class).setText(Integer.toString(folds));
            field(panel, "m_RandomSeedText", JTextField.class).setText(Integer.toString(seed));
            explorer.getTabbedPane().setSelectedComponent(panel);
            int historyAtStart = panel.getResultHistory().getList().getModel().getSize();
            field(panel, "m_StartBut", JButton.class).doClick();
            String operationId = java.util.UUID.randomUUID().toString();
            lastClassifierRun = new ClassifierRun(operationId, ExplorerResultMetadata.snapshot(data),
                    historyAtStart, className, folds, seed);
            ObjectNode result = JSON.createObjectNode();
            result.put("operation", "classifier_training");
            result.put("operation_id", operationId);
            result.put("training_dataset_version", lastClassifierRun.dataset.path("dataset_version").asText());
            result.put("status", "RUNNING");
            result.put("classifier_class", className);
            result.put("folds", folds);
            result.put("seed", seed);
            result.put("classifier_running", running(panel));
            return result;
        }));
    }

    private ObjectNode results(JsonNode arguments) throws Exception {
        int waitSeconds = optionalInt(arguments, "wait_seconds", 0);
        if (waitSeconds < 0 || waitSeconds > 150) {
            throw new IllegalArgumentException("wait_seconds must be between 0 and 150");
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(waitSeconds);
        while (onEdt(() -> running(classifier())) && System.nanoTime() < deadline) {
            Thread.sleep(50);
        }
        return onEdt(() -> {
            ClassifierPanel panel = classifier();
            ResultHistoryPanel history = panel.getResultHistory();
            int count = history.getList().getModel().getSize();
            boolean running = running(panel);
            ClassifierRun run = lastClassifierRun;
            boolean complete = run == null ? !running && count > 0
                    : !running && !run.cancelled && count > run.historyAtStart;
            ObjectNode result = JSON.createObjectNode();
            result.put("classifier_running", running);
            result.put("result_count", count);
            result.put("complete", complete);
            result.put("partial", running);
            if (running && run != null) {
                result.put("name", "");
                result.put("output", field(panel, "m_OutText", JTextArea.class).getText());
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
                result.put("output", field(panel, "m_OutText", JTextArea.class).getText());
                result.put("model_available", false);
            }
            result.set("evaluation_metrics", ExplorerResultMetadata.evaluationMetrics(result.path("output").asText()));
            if (run != null) {
                result.put("classifier_class", run.classifierClass);
                result.put("evaluation_mode", "cross_validation");
                result.put("folds", run.folds);
                result.put("seed", run.seed);
            }
            ExplorerResultMetadata.addProvenance(result, run == null ? "" : run.id,
                    run == null ? null : run.dataset, running, complete,
                    explorer.getPreprocessPanel().getInstances());
            if (run != null && run.cancelled && !running
                    && !result.path("dataset_changed_since_training").booleanValue()) {
                result.put("result_status", "CANCELLED");
            }
            return result;
        });
    }

    private ObjectNode stop() throws Exception {
        return onEdt(() -> {
            ClassifierPanel panel = classifier();
            boolean running = running(panel);
            if (running) {
                if (lastClassifierRun != null) {
                    lastClassifierRun.cancelled = true;
                }
                field(panel, "m_StopBut", AbstractButton.class).doClick();
            }
            ObjectNode result = JSON.createObjectNode();
            result.put("stop_requested", running);
            result.put("classifier_running", running(panel));
            return result;
        });
    }

    public void close() {
        try {
            stop();
        } catch (Exception ignored) {
        }
        clusterOperations.close();
        associationOperations.close();
        attributeOperations.close();
        visualizationOperations.close();
    }

    private static final class ClassifierRun {
        private final String id;
        private final ObjectNode dataset;
        private final int historyAtStart;
        private final String classifierClass;
        private final int folds;
        private final int seed;
        private boolean cancelled;

        private ClassifierRun(String id, ObjectNode dataset, int historyAtStart, String classifierClass,
                int folds, int seed) {
            this.id = id;
            this.dataset = dataset;
            this.historyAtStart = historyAtStart;
            this.classifierClass = classifierClass;
            this.folds = folds;
            this.seed = seed;
        }
    }

    private ClassifierPanel classifier() {
        return explorer.getPanels().stream().filter(ClassifierPanel.class::isInstance)
                .map(ClassifierPanel.class::cast).findFirst()
                .orElseThrow(() -> new IllegalStateException("This Weka version has no Classify panel"));
    }

    private static boolean running(ClassifierPanel panel) throws Exception {
        Thread thread = field(panel, "m_RunThread", Thread.class);
        return thread != null && thread.isAlive();
    }

    private ObjectNode write(Callable<ObjectNode> operation) throws Exception {
        if (!writing.compareAndSet(false, true)) {
            throw new IllegalStateException("Another native Explorer write operation is in progress");
        }
        try {
            return operation.call();
        } finally {
            writing.set(false);
        }
    }

    private static void requireDataset(Instances data) {
        if (data == null || data.numInstances() == 0) {
            throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
        }
    }

    private static int requiredInt(JsonNode arguments, String name) {
        if (!arguments.has(name) || !arguments.get(name).isInt()) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        return arguments.get(name).intValue();
    }

    private static int optionalInt(JsonNode arguments, String name, int defaultValue) {
        if (!arguments.has(name)) {
            return defaultValue;
        }
        if (!arguments.get(name).isInt()) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        return arguments.get(name).intValue();
    }

    private static String optionalText(JsonNode arguments, String name, String defaultValue) {
        if (!arguments.has(name)) {
            return defaultValue;
        }
        if (!arguments.get(name).isTextual() || arguments.get(name).asText().isBlank()) {
            throw new IllegalArgumentException(name + " must be a non-empty string");
        }
        return arguments.get(name).asText();
    }

    static String[] options(JsonNode value) throws Exception {
        if (value == null || value.isNull()) {
            return new String[0];
        }
        if (!value.isArray() || value.size() > 20) {
            throw new IllegalArgumentException("options must be an array with at most 20 entries");
        }
        String[] options = new String[value.size()];
        for (int index = 0; index < options.length; index++) {
            JsonNode option = value.get(index);
            if (!option.isTextual() || option.asText().length() > 64) {
                throw new IllegalArgumentException("options must contain strings of at most 64 characters");
            }
            options[index] = option.asText();
        }
        return options;
    }

    static <T> T field(Object target, String name, Class<T> type) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        Object value = field.get(target);
        if (value == null) {
            return null;
        }
        if (!type.isInstance(value)) {
            throw new IllegalStateException("Unsupported Weka Explorer field " + name + ": " + value.getClass());
        }
        return type.cast(value);
    }

    static <T> T onEdt(Callable<T> action) throws Exception {
        if (SwingUtilities.isEventDispatchThread()) {
            return action.call();
        }
        FutureTask<T> task = new FutureTask<>(action);
        SwingUtilities.invokeAndWait(task);
        try {
            return task.get();
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            if (cause instanceof Exception exception) {
                throw exception;
            }
            throw error;
        }
    }

    static ArrayNode toolSpecifications() {
        ArrayNode tools = JSON.createArrayNode();
        tools.add(specification(STATE, "Read the current Explorer dataset, classifier and activity state", JSON.createObjectNode()));
        tools.add(specification(LOAD, "Load an ARFF or CSV dataset into Explorer Preprocess with an undo point",
                ExplorerDataOperations.loadProperties()));
        tools.add(specification(FILTER, "Apply an enabled Weka filter to the current Explorer dataset with an undo point",
                ExplorerDataOperations.filterProperties()));
        tools.add(specification(DESCRIBE, "Describe the current Explorer dataset with bounded attribute statistics",
                ExplorerDataOperations.describeProperties()));
        tools.add(specification(UNDO, "Undo the latest Explorer Preprocess dataset change and wait for completion",
                JSON.createObjectNode()));
        tools.add(specification(SET_CLASS, "Set the Explorer class index; 0 means the last attribute",
                objectWith("class_index", integer("1-based class index, or 0 for the last attribute"))));
        ObjectNode training = objectWith("classifier_class", string("Enabled Weka classifier class; defaults to J48"),
                "folds", integer("Cross-validation folds, 2 to 50; defaults to 10"),
                "seed", integer("Evaluation random seed; defaults to 1"));
        ObjectNode options = training.putObject("options");
        options.put("type", "array");
        options.set("items", string("Weka classifier option"));
        tools.add(specification(TRAIN, "Start a real Explorer cross-validation task in the background", training));
        tools.add(specification(RESULTS, "Read the latest Explorer result, optionally waiting for completion",
                objectWith("wait_seconds", integer("Seconds to wait for the running task, 0 to 150"))));
        tools.add(specification(STOP, "Request cancellation of the current Explorer classifier task", JSON.createObjectNode()));
        tools.add(specification(CLUSTER, "Start a real Explorer clustering task in the background",
                ExplorerClusterOperations.trainProperties()));
        tools.add(specification(CLUSTER_RESULTS, "Read the latest Explorer clustering result, optionally waiting for completion",
                ExplorerClusterOperations.resultProperties()));
        tools.add(specification(CLUSTER_STOP, "Request cancellation of the current Explorer clustering task",
                JSON.createObjectNode()));
        tools.add(specification(ASSOCIATE, "Start association rule mining in the Explorer Associate panel",
                ExplorerAssociationOperations.trainProperties()));
        tools.add(specification(ASSOCIATE_RESULTS, "Read the latest association result, optionally waiting for completion",
                ExplorerAssociationOperations.resultProperties()));
        tools.add(specification(ASSOCIATE_STOP, "Request cancellation of the current association task",
                JSON.createObjectNode()));
        tools.add(specification(SELECT_ATTRIBUTES, "Start attribute evaluation in the Explorer Select attributes panel",
                ExplorerAttributeSelectionOperations.selectionProperties()));
        tools.add(specification(SELECT_RESULTS, "Read the latest attribute selection result, optionally waiting for completion",
                ExplorerAttributeSelectionOperations.resultProperties()));
        tools.add(specification(SELECT_STOP, "Request cancellation of the current attribute selection task",
                JSON.createObjectNode()));
        tools.add(specification(VISUALIZE, "Open a native Weka scatter plot for the current Explorer dataset",
                ExplorerVisualizationOperations.visualizeProperties()));
        tools.add(specification(CAPTURE_VISUALIZATION, "Capture the current native scatter plot as PNG image content",
                ExplorerVisualizationOperations.captureProperties()));
        tools.add(specification(CLOSE_VISUALIZATION, "Close only the visualization window created by this bridge",
                JSON.createObjectNode()));
        return tools;
    }

    private static ObjectNode specification(String name, String description, ObjectNode properties) {
        ObjectNode tool = JSON.createObjectNode();
        tool.put("name", name);
        tool.put("description", description);
        ObjectNode schema = JSON.createObjectNode();
        schema.put("type", "object");
        schema.set("properties", properties);
        tool.set("inputSchema", schema);
        return tool;
    }

    private static ObjectNode objectWith(String first, JsonNode firstValue, String second, JsonNode secondValue,
            String third, JsonNode thirdValue) {
        ObjectNode result = objectWith(first, firstValue, second, secondValue);
        result.set(third, thirdValue);
        return result;
    }

    private static ObjectNode objectWith(String first, JsonNode firstValue, String second, JsonNode secondValue) {
        ObjectNode result = objectWith(first, firstValue);
        result.set(second, secondValue);
        return result;
    }

    private static ObjectNode objectWith(String name, JsonNode value) {
        ObjectNode result = JSON.createObjectNode();
        result.set(name, value);
        return result;
    }

    private static ObjectNode integer(String description) {
        ObjectNode result = JSON.createObjectNode();
        result.put("type", "integer");
        result.put("description", description);
        return result;
    }

    private static ObjectNode string(String description) {
        ObjectNode result = JSON.createObjectNode();
        result.put("type", "string");
        result.put("description", description);
        return result;
    }
}
