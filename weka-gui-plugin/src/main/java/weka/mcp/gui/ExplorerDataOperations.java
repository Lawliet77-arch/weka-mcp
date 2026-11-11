package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.gui.explorer.Explorer;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;

final class ExplorerDataOperations {
    private static final JsonMapper JSON = new JsonMapper();
    private static final Set<String> FILTERS = Set.of(
            "weka.filters.unsupervised.attribute.ReplaceMissingValues",
            "weka.filters.unsupervised.attribute.Normalize",
            "weka.filters.unsupervised.attribute.Standardize",
            "weka.filters.unsupervised.attribute.Discretize",
            "weka.filters.supervised.attribute.Discretize");
    private final Explorer explorer;
    private final PluginWorkspace workspace;

    ExplorerDataOperations(Explorer explorer, PluginWorkspace workspace) {
        this.explorer = explorer;
        this.workspace = workspace;
    }

    ObjectNode load(JsonNode arguments) throws Exception {
        if (workspace == null) {
            throw new IllegalStateException("Dataset import is not configured");
        }
        if (!arguments.path("path").isTextual() || arguments.path("path").asText().isBlank()) {
            throw new IllegalArgumentException("path must be a non-empty ARFF or CSV path");
        }
        Path supplied = Path.of(arguments.path("path").asText());
        if (!supplied.isAbsolute() && !supplied.normalize().startsWith(workspace.workspaceDirectory()
                .resolve("data").getFileName())) {
            supplied = workspace.workspaceDirectory().resolve("data").resolve(supplied);
        }
        String imported = workspace.importDataset(supplied);
        Path dataset = workspace.resolveDataset(imported);
        Instances data = loadDataset(dataset);
        if (data.numInstances() == 0) {
            throw new IllegalArgumentException("Dataset must contain at least one instance");
        }
        if (arguments.has("class_index") && !arguments.get("class_index").isInt()) {
            throw new IllegalArgumentException("class_index must be an integer");
        }
        int requested = arguments.has("class_index") ? arguments.get("class_index").intValue() : 0;
        int classIndex = requested == 0 ? data.numAttributes() - 1 : requested - 1;
        if (classIndex < 0 || classIndex >= data.numAttributes()) {
            throw new IllegalArgumentException("class_index must be 0 (last) or between 1 and " + data.numAttributes());
        }
        data.setClassIndex(classIndex);
        return ExplorerSession.onEdt(() -> {
            ensureIdle();
            explorer.getPreprocessPanel().addUndoPoint();
            explorer.getPreprocessPanel().setInstances(data);
            explorer.getTabbedPane().setSelectedIndex(0);
            ObjectNode result = new ExplorerSession(explorer).state();
            result.put("loaded", true);
            result.put("source", imported);
            return result;
        });
    }

    static Instances loadDataset(Path dataset) throws Exception {
        String filename = dataset.getFileName().toString().toLowerCase(Locale.ROOT);
        AbstractFileLoader loader;
        if (filename.endsWith(".arff")) {
            loader = new ArffLoader();
        } else if (filename.endsWith(".csv")) {
            loader = new CSVLoader();
        } else {
            throw new IllegalArgumentException("Dataset must be an ARFF or CSV file");
        }
        try (InputStream input = Files.newInputStream(dataset)) {
            loader.setSource(input);
            return loader.getDataSet();
        }
    }

    ObjectNode filter(JsonNode arguments) throws Exception {
        String className = arguments.has("filter_class")
                ? arguments.path("filter_class").asText() : "weka.filters.unsupervised.attribute.ReplaceMissingValues";
        if (!FILTERS.contains(className)) {
            throw new IllegalArgumentException("Filter is not enabled for native Explorer automation: "
                    + className + "; enabled=" + FILTERS);
        }
        Filter filter = (Filter) Class.forName(className).getDeclaredConstructor().newInstance();
        String[] options = ExplorerSession.options(arguments.get("options"));
        filter.setOptions(options);
        Object[] snapshot = ExplorerSession.onEdt(() -> {
            Instances current = explorer.getPreprocessPanel().getInstances();
            if (current == null || current.numInstances() == 0) {
                throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
            }
            return new Object[]{current, new Instances(current)};
        });
        Instances original = (Instances) snapshot[0];
        filter.setInputFormat((Instances) snapshot[1]);
        Instances filtered = Filter.useFilter((Instances) snapshot[1], filter);
        return ExplorerSession.onEdt(() -> {
            ensureIdle();
            if (explorer.getPreprocessPanel().getInstances() != original) {
                throw new IllegalStateException("Explorer data changed while filtering; no new data was applied");
            }
            explorer.getPreprocessPanel().addUndoPoint();
            explorer.getPreprocessPanel().setInstances(filtered);
            ObjectNode result = JSON.createObjectNode();
            result.put("filtered", true);
            result.put("filter_class", className);
            result.put("instances", filtered.numInstances());
            result.put("attributes", filtered.numAttributes());
            result.put("undo_available", true);
            result.put("evaluation_warning", "The filter was fitted on the full Explorer dataset before evaluation");
            return result;
        });
    }

    ObjectNode describe(JsonNode arguments) throws Exception {
        int maxAttributes = bounded(arguments, "max_attributes", 100, 1, 1000);
        int maxNominalValues = bounded(arguments, "max_nominal_values", 20, 0, 100);
        Instances data = ExplorerSession.onEdt(() -> {
            Instances current = explorer.getPreprocessPanel().getInstances();
            if (current == null || current.numInstances() == 0) {
                throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
            }
            return new Instances(current);
        });
        ObjectNode result = JSON.createObjectNode();
        result.put("statistics_scope", "snapshot");
        result.put("relation_name", data.relationName());
        result.put("instances", data.numInstances());
        result.put("sum_of_weights", data.sumOfWeights());
        result.put("attributes_total", data.numAttributes());
        result.put("class_index", data.classIndex() < 0 ? 0 : data.classIndex() + 1);
        result.put("class_attribute", data.classIndex() < 0 ? "" : data.classAttribute().name());
        int count = Math.min(maxAttributes, data.numAttributes());
        result.put("attributes_returned", count);
        result.put("truncated", count < data.numAttributes());
        long missing = 0;
        for (int index = 0; index < data.numAttributes(); index++) {
            missing += data.attributeStats(index).missingCount;
        }
        result.put("missing_total", missing);
        ArrayNode attributes = result.putArray("attributes");
        for (int index = 0; index < count; index++) {
            Attribute attribute = data.attribute(index);
            AttributeStats stats = data.attributeStats(index);
            ObjectNode item = attributes.addObject();
            item.put("index", index + 1);
            item.put("name", attribute.name());
            item.put("type", type(attribute));
            item.put("class", index == data.classIndex());
            item.put("missing", stats.missingCount);
            item.put("distinct", stats.distinctCount);
            item.put("unique", stats.uniqueCount);
            if (attribute.isNumeric()) {
                item.put("min", stats.numericStats.min);
                item.put("max", stats.numericStats.max);
                item.put("mean", stats.numericStats.mean);
                item.put("stddev", stats.numericStats.stdDev);
            }
            if (attribute.isNominal()) {
                ArrayNode values = item.putArray("values");
                for (int value = 0; value < Math.min(attribute.numValues(), maxNominalValues); value++) {
                    values.add(attribute.value(value));
                }
                item.put("values_truncated", attribute.numValues() > maxNominalValues);
            }
        }
        return result;
    }

    ObjectNode undo() throws Exception {
        if (new ExplorerSession(explorer).state().get("task_running").booleanValue()) {
            throw new IllegalStateException("Stop native Explorer tasks before undoing dataset changes");
        }
        Instances before = ExplorerSession.onEdt(() -> {
            Thread running = ExplorerSession.field(explorer.getPreprocessPanel(), "m_IOThread", Thread.class);
            if (running != null && running.isAlive()) {
                throw new IllegalStateException("Explorer Preprocess is busy with another I/O operation");
            }
            Instances current = explorer.getPreprocessPanel().getInstances();
            explorer.getPreprocessPanel().undo();
            return current;
        });
        Thread loading = ExplorerSession.onEdt(() ->
                ExplorerSession.field(explorer.getPreprocessPanel(), "m_IOThread", Thread.class));
        if (loading != null) {
            loading.join(30_000);
            if (loading.isAlive()) {
                throw new IllegalStateException("Timed out waiting for Explorer undo to finish");
            }
        }
        ExplorerSession.onEdt(() -> null);
        ObjectNode result = new ExplorerSession(explorer).state();
        result.put("changed", before != explorer.getPreprocessPanel().getInstances());
        result.put("scope", "preprocess_dataset");
        if (result.path("result_count").intValue() > 0) {
            result.put("existing_results_may_be_stale", true);
        }
        return result;
    }

    private void ensureIdle() throws Exception {
        if (new ExplorerSession(explorer).state().path("task_running").booleanValue()) {
            throw new IllegalStateException("Stop the native Explorer task before changing the dataset");
        }
    }

    private static int bounded(JsonNode arguments, String name, int defaultValue, int minimum, int maximum) {
        int value = arguments.has(name) ? arguments.path(name).intValue() : defaultValue;
        if (arguments.has(name) && !arguments.path(name).isInt()) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        if (value < minimum || value > maximum) {
            throw new IllegalArgumentException(name + " must be between " + minimum + " and " + maximum);
        }
        return value;
    }

    private static String type(Attribute attribute) {
        return switch (attribute.type()) {
            case Attribute.NUMERIC -> "numeric";
            case Attribute.NOMINAL -> "nominal";
            case Attribute.STRING -> "string";
            case Attribute.DATE -> "date";
            case Attribute.RELATIONAL -> "relational";
            default -> "unknown";
        };
    }

    static ObjectNode describeProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode attributes = properties.putObject("max_attributes");
        attributes.put("type", "integer");
        attributes.put("description", "Maximum attributes returned, 1 to 1000; defaults to 100");
        ObjectNode values = properties.putObject("max_nominal_values");
        values.put("type", "integer");
        values.put("description", "Maximum nominal values per attribute, 0 to 100; defaults to 20");
        return properties;
    }

    static ObjectNode loadProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode path = properties.putObject("path");
        path.put("type", "string");
        path.put("description", "ARFF/CSV path inside the plugin data directory, or a trusted local file");
        ObjectNode classIndex = properties.putObject("class_index");
        classIndex.put("type", "integer");
        classIndex.put("description", "1-based class index; 0 selects the last attribute");
        return properties;
    }

    static ObjectNode filterProperties() {
        ObjectNode properties = JSON.createObjectNode();
        ObjectNode filter = properties.putObject("filter_class");
        filter.put("type", "string");
        filter.put("description", "Enabled Weka filter; defaults to ReplaceMissingValues");
        ObjectNode options = properties.putObject("options");
        options.put("type", "array");
        ObjectNode items = options.putObject("items");
        items.put("type", "string");
        items.put("description", "Weka filter option");
        return properties;
    }
}
