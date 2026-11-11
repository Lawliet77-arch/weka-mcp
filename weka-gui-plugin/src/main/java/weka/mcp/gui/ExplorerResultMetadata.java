package weka.mcp.gui;

import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Instances;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class ExplorerResultMetadata {
    private static final JsonMapper JSON = new JsonMapper();
    private static final String NUMBER = "([-+]?\\d+(?:\\.\\d*)?(?:[eE][-+]?\\d+)?)";
    private static final Map<String, String> SUMMARY_FIELDS = new LinkedHashMap<>();

    static {
        SUMMARY_FIELDS.put("Correctly Classified Instances", "correctly_classified");
        SUMMARY_FIELDS.put("Incorrectly Classified Instances", "incorrectly_classified");
        SUMMARY_FIELDS.put("Kappa statistic", "kappa");
        SUMMARY_FIELDS.put("Mean absolute error", "mean_absolute_error");
        SUMMARY_FIELDS.put("Root mean squared error", "root_mean_squared_error");
        SUMMARY_FIELDS.put("Relative absolute error", "relative_absolute_error_percent");
        SUMMARY_FIELDS.put("Root relative squared error", "root_relative_squared_error_percent");
        SUMMARY_FIELDS.put("Total Number of Instances", "total_instances");
    }

    private ExplorerResultMetadata() {
    }

    static ObjectNode snapshot(Instances data) {
        ObjectNode result = JSON.createObjectNode();
        result.put("dataset_version", version(data));
        result.put("relation_name", data == null ? "" : data.relationName());
        result.put("instances", data == null ? 0 : data.numInstances());
        result.put("attributes", data == null ? 0 : data.numAttributes());
        result.put("class_index", data == null || data.classIndex() < 0 ? 0 : data.classIndex() + 1);
        result.put("class_attribute", data == null || data.classIndex() < 0 ? "" : data.classAttribute().name());
        return result;
    }

    static String version(Instances data) {
        return data == null ? "" : "local-" + Integer.toHexString(System.identityHashCode(data));
    }

    static void addProvenance(ObjectNode result, String operationId, ObjectNode trainingDataset,
            boolean running, boolean complete, Instances currentDataset) {
        if (operationId != null && !operationId.isBlank()) {
            result.put("operation_id", operationId);
        }
        if (trainingDataset != null) {
            result.set("training_dataset", trainingDataset);
        }
        result.set("current_dataset", snapshot(currentDataset));
        if (trainingDataset == null) {
            result.putNull("dataset_changed_since_training");
            result.put("result_status", "UNTRACKED_HISTORY");
        } else {
            boolean changed = !Objects.equals(trainingDataset.path("dataset_version").asText(), version(currentDataset));
            result.put("dataset_changed_since_training", changed);
            result.put("result_status", changed ? "STALE_DATASET"
                    : running ? "RUNNING" : complete ? "COMPLETE" : "INCOMPLETE");
        }
    }

    static ObjectNode evaluationMetrics(String output) {
        ObjectNode metrics = JSON.createObjectNode();
        if (output == null) {
            return metrics;
        }
        for (Map.Entry<String, String> entry : SUMMARY_FIELDS.entrySet()) {
            Matcher matcher = Pattern.compile("(?m)^" + Pattern.quote(entry.getKey()) + "\\s+" + NUMBER
                    + "(?:\\s+" + NUMBER + ")?\\s*(%)?\\s*$").matcher(output);
            if (matcher.find()) {
                metrics.put(entry.getValue(), Double.parseDouble(matcher.group(1)));
                if (entry.getKey().endsWith("Classified Instances") && matcher.group(3) != null) {
                    metrics.put(entry.getValue() + "_percent", Double.parseDouble(
                            matcher.group(2) == null ? matcher.group(1) : matcher.group(2)));
                }
            }
        }
        Matcher cluster = Pattern.compile("Within cluster sum of squared errors:\\s*" + NUMBER).matcher(output);
        if (cluster.find()) {
            metrics.put("within_cluster_squared_error", Double.parseDouble(cluster.group(1)));
        }
        return metrics;
    }
}
