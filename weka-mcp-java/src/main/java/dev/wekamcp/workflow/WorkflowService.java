package dev.wekamcp.workflow;

import dev.wekamcp.config.ServerConfig;
import dev.wekamcp.config.Whitelist;
import dev.wekamcp.engine.TaskRunner;
import dev.wekamcp.registry.SessionRegistry;
import dev.wekamcp.registry.SessionRegistry.DatasetEntry;
import dev.wekamcp.tools.ToolSupport;
import dev.wekamcp.tools.WekaTools;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class WorkflowService {
    private final ServerConfig config;
    private final Whitelist whitelist;
    private final SessionRegistry registry;
    private final TaskRunner tasks;

    public WorkflowService(ServerConfig config, Whitelist whitelist, SessionRegistry registry, TaskRunner tasks) {
        this.config = config;
        this.whitelist = whitelist;
        this.registry = registry;
        this.tasks = tasks;
    }

    public ObjectNode split(Map<String, Object> arguments) {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        if (registry.datasets().size() + 2 > config.maxDatasets()) {
            throw new IllegalStateException("Split requires two free dataset slots; release an unused dataset first");
        }
        Instances data = WekaTools.classificationData(entry, ToolSupport.optionalInt(arguments, "class_index", 0));
        double trainPercent = ToolSupport.optionalDouble(arguments, "train_percent", 80.0);
        long seed = ToolSupport.optionalLong(arguments, "seed", 1L);
        if (trainPercent <= 0 || trainPercent >= 100) {
            throw new IllegalArgumentException("train_percent must be between 0 and 100");
        }
        WekaTools.DataSplit split = WekaTools.splitData(data, trainPercent, seed);
        String prefix = ToolSupport.optionalString(arguments, "name_prefix", entry.name());
        DatasetEntry training = registry.registerDataset(prefix + ".train", split.training());
        DatasetEntry test = registry.registerDataset(prefix + ".test", split.test());
        ObjectNode result = ToolSupport.object();
        result.put("source_dataset_id", entry.id());
        result.put("training_dataset_id", training.id());
        result.put("test_dataset_id", test.id());
        result.put("training_instances", training.data().numInstances());
        result.put("test_instances", test.data().numInstances());
        ToolSupport.putNumber(result, "train_percent", trainPercent);
        result.put("seed", seed);
        result.put("class_index", data.classIndex() + 1);
        result.put("stratified", data.classAttribute().isNominal());
        return result;
    }

    public ObjectNode preview(Map<String, Object> arguments) throws Exception {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        String className = ToolSupport.requiredString(arguments, "filter_class");
        String[] options = ToolSupport.options(arguments, config.maxOptionsLength(), config.maxOptionTokenLength());
        Filter filter = (Filter) whitelist.create(Whitelist.Kind.FILTER, className, options);
        Instances input = new Instances(entry.data());
        int classIndex = ToolSupport.optionalInt(arguments, "class_index", 0);
        if (classIndex < 0 || classIndex > input.numAttributes()) {
            throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
        }
        if (classIndex > 0) {
            input.setClassIndex(classIndex - 1);
        }
        if (filter instanceof SupervisedFilter && input.classIndex() < 0) {
            throw new IllegalArgumentException("A supervised filter requires class_index");
        }
        int maximum = ToolSupport.optionalInt(arguments, "max_rows", 10);
        if (maximum < 0 || maximum > config.maxPredictionRows()) {
            throw new IllegalArgumentException("max_rows must be between 0 and maxPredictionRows=" + config.maxPredictionRows());
        }
        Instances output = tasks.call(config.loadTimeoutSeconds(), () -> {
            filter.setInputFormat(input);
            return Filter.useFilter(input, filter);
        });
        if (output.numAttributes() > config.maxAttributes()) {
            throw new IllegalArgumentException("Transformed attribute count exceeds maxAttributes=" + config.maxAttributes());
        }
        ObjectNode result = ToolSupport.object();
        result.put("source_dataset_id", entry.id());
        result.put("filter_class", className);
        result.put("fit_scope", "full_dataset");
        result.put("fold_safe", false);
        result.put("warning", "Preview filters are fitted on the full dataset; use fold-safe preprocessing for evaluation");
        result.put("instances", output.numInstances());
        result.put("attributes", output.numAttributes());
        result.put("class_index", output.classIndex() + 1);
        ArrayNode attributes = result.putArray("attribute_names");
        for (int index = 0; index < output.numAttributes(); index++) {
            attributes.add(output.attribute(index).name());
        }
        ArrayNode rows = result.putArray("rows");
        for (int index = 0; index < Math.min(maximum, output.numInstances()); index++) {
            ArrayNode row = rows.addArray();
            Instance instance = output.instance(index);
            for (int attribute = 0; attribute < output.numAttributes(); attribute++) {
                addValue(row, instance, attribute);
            }
        }
        return result;
    }

    public ObjectNode histogram(Map<String, Object> arguments) {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        Instances data = entry.data();
        int index = requiredAttribute(arguments, "attribute_index", data);
        int bins = ToolSupport.optionalInt(arguments, "bins", 10);
        if (bins < 1 || bins > 100) {
            throw new IllegalArgumentException("bins must be between 1 and 100");
        }
        Attribute attribute = data.attribute(index);
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("attribute_index", index + 1);
        result.put("attribute", attribute.name());
        result.put("type", attribute.isNominal() ? "nominal" : "numeric");
        int missing = 0;
        int valid = 0;
        double minimum = Double.POSITIVE_INFINITY;
        double maximum = Double.NEGATIVE_INFINITY;
        double mean = 0;
        for (Instance instance : data) {
            if (instance.isMissing(index)) {
                missing++;
            } else {
                double value = instance.value(index);
                if (attribute.isNumeric()) {
                    minimum = Math.min(minimum, value);
                    maximum = Math.max(maximum, value);
                    mean += value;
                }
                valid++;
            }
        }
        result.put("missing", missing);
        result.put("valid", valid);
        ArrayNode output = result.putArray("bins");
        if (attribute.isNominal()) {
            for (int value = 0; value < attribute.numValues(); value++) {
                int count = 0;
                for (Instance instance : data) {
                    if (!instance.isMissing(index) && (int) instance.value(index) == value) {
                        count++;
                    }
                }
                ObjectNode bin = output.addObject();
                bin.put("label", attribute.value(value));
                bin.put("count", count);
                ToolSupport.putNumber(bin, "percent", percent(count, valid));
            }
        } else if (attribute.isNumeric()) {
            if (valid > 0) {
                mean /= valid;
            }
            ToolSupport.putNumber(result, "min", minimum);
            ToolSupport.putNumber(result, "max", maximum);
            ToolSupport.putNumber(result, "mean", mean);
            int actualBins = minimum == maximum ? 1 : bins;
            int[] counts = new int[actualBins];
            for (Instance instance : data) {
                if (!instance.isMissing(index)) {
                    double fraction = (instance.value(index) - minimum) / (maximum - minimum);
                    int bin = minimum == maximum ? 0 : Math.min(actualBins - 1, (int) (fraction * actualBins));
                    counts[bin]++;
                }
            }
            for (int indexBin = 0; indexBin < counts.length; indexBin++) {
                double lower = minimum + (maximum - minimum) * indexBin / actualBins;
                double upper = minimum + (maximum - minimum) * (indexBin + 1) / actualBins;
                ObjectNode bin = output.addObject();
                ToolSupport.putNumber(bin, "lower", lower);
                ToolSupport.putNumber(bin, "upper", upper);
                bin.put("count", counts[indexBin]);
                ToolSupport.putNumber(bin, "percent", percent(counts[indexBin], valid));
            }
        } else {
            throw new IllegalArgumentException("histogram supports numeric and nominal attributes");
        }
        return result;
    }

    public ObjectNode scatter(Map<String, Object> arguments) {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        Instances data = entry.data();
        int x = requiredAttribute(arguments, "x_index", data);
        int y = requiredAttribute(arguments, "y_index", data);
        if (!data.attribute(x).isNumeric() || !data.attribute(y).isNumeric()) {
            throw new IllegalArgumentException("scatter requires two numeric attributes");
        }
        int classIndex = ToolSupport.optionalInt(arguments, "class_index", 0);
        if (classIndex < 0 || classIndex > data.numAttributes()) {
            throw new IllegalArgumentException("class_index is one-based and must identify an existing attribute");
        }
        int color = classIndex == 0 ? data.classIndex() : classIndex - 1;
        int maximum = ToolSupport.optionalInt(arguments, "max_points", Math.min(500, config.maxPredictionRows()));
        if (maximum < 1 || maximum > config.maxPredictionRows()) {
            throw new IllegalArgumentException("max_points must be between 1 and maxPredictionRows=" + config.maxPredictionRows());
        }
        int stride = Math.max(1, (data.numInstances() + maximum - 1) / maximum);
        ArrayNode points = ToolSupport.array();
        int skipped = 0;
        for (int index = 0; index < data.numInstances(); index += stride) {
            Instance instance = data.instance(index);
            if (instance.isMissing(x) || instance.isMissing(y)) {
                skipped++;
                continue;
            }
            ObjectNode point = points.addObject();
            point.put("source_index", index);
            ToolSupport.putNumber(point, "x", instance.value(x));
            ToolSupport.putNumber(point, "y", instance.value(y));
            if (color >= 0) {
                if (instance.isMissing(color)) {
                    point.putNull("class");
                } else if (data.classAttribute().isNominal()) {
                    point.put("class", instance.stringValue(color));
                } else {
                    ToolSupport.putNumber(point, "class", instance.value(color));
                }
            }
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("x_attribute", data.attribute(x).name());
        result.put("y_attribute", data.attribute(y).name());
        result.put("class_index", color + 1);
        result.put("count", points.size());
        result.put("sampled", points.size() < data.numInstances());
        result.put("skipped_missing", skipped);
        result.set("points", points);
        return result;
    }

    public ObjectNode scatterMatrix(Map<String, Object> arguments) {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        Instances data = entry.data();
        List<Integer> indices = new ArrayList<>();
        Object requested = arguments.get("attributes");
        if (requested == null) {
            for (int index = 0; index < data.numAttributes() && indices.size() < 5; index++) {
                if (data.attribute(index).isNumeric()) {
                    indices.add(index);
                }
            }
        } else {
            if (!(requested instanceof List<?> attributes)) {
                throw new IllegalArgumentException("attributes must be an array of one-based numeric indices");
            }
            for (Object value : attributes) {
                if (!(value instanceof Number number) || number.doubleValue() != number.longValue()) {
                    throw new IllegalArgumentException("attributes must contain only integer indices");
                }
                int index = number.intValue() - 1;
                if (index < 0 || index >= data.numAttributes() || !data.attribute(index).isNumeric()) {
                    throw new IllegalArgumentException("scatter matrix attributes must identify numeric attributes");
                }
                indices.add(index);
            }
        }
        if (indices.size() < 2 || indices.size() > 5 || new LinkedHashSet<>(indices).size() != indices.size()) {
            throw new IllegalArgumentException("attributes must contain between 2 and 5 distinct numeric attributes");
        }
        int maximum = ToolSupport.optionalInt(arguments, "max_points", 100);
        ArrayNode plots = ToolSupport.array();
        for (int left = 0; left < indices.size(); left++) {
            for (int right = left + 1; right < indices.size(); right++) {
                Map<String, Object> pair = new LinkedHashMap<>();
                pair.put("dataset_id", entry.id());
                pair.put("x_index", indices.get(left) + 1);
                pair.put("y_index", indices.get(right) + 1);
                pair.put("max_points", maximum);
                if (arguments.containsKey("class_index")) {
                    pair.put("class_index", arguments.get("class_index"));
                }
                plots.add(scatter(pair));
            }
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("plot_count", plots.size());
        result.put("max_points", maximum);
        result.set("attribute_indices", ToolSupport.JSON.valueToTree(indices.stream().map(index -> index + 1).toList()));
        result.set("plots", plots);
        return result;
    }

    public ObjectNode compare(Map<String, Object> arguments) {
        DatasetEntry left = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        DatasetEntry right = registry.dataset(ToolSupport.requiredString(arguments, "other_dataset_id"));
        int bins = ToolSupport.optionalInt(arguments, "bins", 10);
        int maximum = ToolSupport.optionalInt(arguments, "max_attributes", 20);
        if (bins < 1 || bins > 100 || maximum < 1 || maximum > 100) {
            throw new IllegalArgumentException("bins must be 1..100 and max_attributes must be 1..100");
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", left.id());
        result.put("other_dataset_id", right.id());
        result.put("same_header", left.data().equalHeaders(right.data()));
        Set<String> rightNames = new LinkedHashSet<>();
        for (int index = 0; index < right.data().numAttributes(); index++) {
            rightNames.add(right.data().attribute(index).name());
        }
        Set<String> leftNames = new LinkedHashSet<>();
        ArrayNode onlyLeft = result.putArray("only_left");
        ArrayNode onlyRight = result.putArray("only_right");
        ArrayNode attributes = result.putArray("attributes");
        for (int index = 0; index < left.data().numAttributes(); index++) {
            Attribute leftAttribute = left.data().attribute(index);
            leftNames.add(leftAttribute.name());
            Attribute rightAttribute = right.data().attribute(leftAttribute.name());
            if (rightAttribute == null) {
                onlyLeft.add(leftAttribute.name());
            } else if (attributes.size() < maximum) {
                attributes.add(attributeDrift(left.data(), right.data(), index,
                        attributeIndex(right.data(), rightAttribute.name()), bins));
            }
        }
        for (int index = 0; index < right.data().numAttributes(); index++) {
            String name = right.data().attribute(index).name();
            if (!leftNames.contains(name)) {
                onlyRight.add(name);
            }
        }
        result.put("truncated", attributes.size() < left.data().numAttributes());
        result.put("psi_note", "PSI is a descriptive drift statistic; low < 0.1, moderate < 0.25, high >= 0.25, not a significance test");
        return result;
    }

    private static ObjectNode attributeDrift(Instances left, Instances right, int leftIndex, int rightIndex, int bins) {
        Attribute attribute = left.attribute(leftIndex);
        ObjectNode result = ToolSupport.object();
        result.put("attribute", attribute.name());
        result.put("type_compatible", attribute.type() == right.attribute(rightIndex).type());
        if (attribute.type() != right.attribute(rightIndex).type()) {
            return result;
        }
        if (attribute.isNominal()) {
            Set<String> labels = new LinkedHashSet<>();
            for (int value = 0; value < attribute.numValues(); value++) {
                labels.add(attribute.value(value));
            }
            for (Instance instance : left) {
                labels.add(instance.isMissing(leftIndex) ? "__MISSING__" : instance.stringValue(leftIndex));
            }
            for (Instance instance : right) {
                labels.add(instance.isMissing(rightIndex) ? "__MISSING__" : instance.stringValue(rightIndex));
            }
            Map<String, Integer> leftCounts = new LinkedHashMap<>();
            Map<String, Integer> rightCounts = new LinkedHashMap<>();
            for (String label : labels) {
                leftCounts.put(label, 0);
                rightCounts.put(label, 0);
            }
            for (Instance instance : left) {
                String label = instance.isMissing(leftIndex) ? "__MISSING__" : instance.stringValue(leftIndex);
                leftCounts.compute(label, (key, value) -> value == null ? 1 : value + 1);
            }
            for (Instance instance : right) {
                String label = instance.isMissing(rightIndex) ? "__MISSING__" : instance.stringValue(rightIndex);
                rightCounts.compute(label, (key, value) -> value == null ? 1 : value + 1);
            }
            double psi = 0;
            ArrayNode distributions = result.putArray("distribution");
            for (String label : labels) {
                double p = (leftCounts.get(label) + 1e-6) / (left.numInstances() + 1e-6 * labels.size());
                double q = (rightCounts.get(label) + 1e-6) / (right.numInstances() + 1e-6 * labels.size());
                double contribution = (p - q) * Math.log(p / q);
                psi += contribution;
                ObjectNode distribution = distributions.addObject();
                distribution.put("label", label);
                distribution.put("left", leftCounts.get(label));
                distribution.put("right", rightCounts.get(label));
                ToolSupport.putNumber(distribution, "psi_contribution", contribution);
            }
            ToolSupport.putNumber(result, "psi", psi);
            result.put("drift", psi < 0.1 ? "low" : psi < 0.25 ? "moderate" : "high");
        } else if (attribute.isNumeric()) {
            double minimum = Double.POSITIVE_INFINITY;
            double maximum = Double.NEGATIVE_INFINITY;
            for (Instance instance : left) {
                if (!instance.isMissing(leftIndex)) {
                    minimum = Math.min(minimum, instance.value(leftIndex));
                    maximum = Math.max(maximum, instance.value(leftIndex));
                }
            }
            for (Instance instance : right) {
                if (!instance.isMissing(rightIndex)) {
                    minimum = Math.min(minimum, instance.value(rightIndex));
                    maximum = Math.max(maximum, instance.value(rightIndex));
                }
            }
            int[] leftCounts = histogramCounts(left, leftIndex, minimum, maximum, bins);
            int[] rightCounts = histogramCounts(right, rightIndex, minimum, maximum, bins);
            double psi = 0;
            for (int index = 0; index < leftCounts.length; index++) {
                double p = (leftCounts[index] + 1e-6) / (left.numInstances() + 1e-6 * leftCounts.length);
                double q = (rightCounts[index] + 1e-6) / (right.numInstances() + 1e-6 * rightCounts.length);
                psi += (p - q) * Math.log(p / q);
            }
            ToolSupport.putNumber(result, "psi", psi);
            result.put("drift", psi < 0.1 ? "low" : psi < 0.25 ? "moderate" : "high");
            ToolSupport.putNumber(result, "min", minimum);
            ToolSupport.putNumber(result, "max", maximum);
            ToolSupport.putNumber(result, "mean_left", mean(left, leftIndex));
            ToolSupport.putNumber(result, "mean_right", mean(right, rightIndex));
        }
        ToolSupport.putNumber(result, "missing_left", missingPercent(left, leftIndex));
        ToolSupport.putNumber(result, "missing_right", missingPercent(right, rightIndex));
        return result;
    }

    private static int[] histogramCounts(Instances data, int index, double minimum, double maximum, int bins) {
        int[] counts = new int[bins + 1];
        for (Instance instance : data) {
            if (instance.isMissing(index)) {
                counts[bins]++;
            } else if (minimum == maximum) {
                counts[0]++;
            } else {
                int bin = Math.max(0, Math.min(bins - 1,
                        (int) ((instance.value(index) - minimum) / (maximum - minimum) * bins)));
                counts[bin]++;
            }
        }
        return counts;
    }

    public ObjectNode export(Map<String, Object> arguments) throws Exception {
        DatasetEntry entry = registry.dataset(ToolSupport.requiredString(arguments, "dataset_id"));
        String format = ToolSupport.optionalString(arguments, "format", "arff").toLowerCase(Locale.ROOT);
        if (!Set.of("arff", "csv").contains(format)) {
            throw new IllegalArgumentException("format must be arff or csv");
        }
        Path target = config.resolveOutputPath(ToolSupport.requiredString(arguments, "path"), "." + format);
        boolean overwrite = ToolSupport.optionalBoolean(arguments, "overwrite", false);
        if (Files.exists(target) && !overwrite) {
            throw new IllegalArgumentException("Output already exists; set overwrite=true to replace it");
        }
        Path temporary = Files.createTempFile(target.getParent(), ".weka-dataset-", ".tmp");
        try {
            if ("arff".equals(format)) {
                ArffSaver saver = new ArffSaver();
                saver.setInstances(entry.data());
                saver.setFile(temporary.toFile());
                saver.writeBatch();
            } else {
                CSVSaver saver = new CSVSaver();
                saver.setInstances(entry.data());
                saver.setFile(temporary.toFile());
                saver.writeBatch();
            }
            if (overwrite) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } else {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (Exception e) {
            Files.deleteIfExists(temporary);
            throw e;
        }
        ObjectNode result = ToolSupport.object();
        result.put("dataset_id", entry.id());
        result.put("path", config.dataRoot().relativize(target).toString().replace('\\', '/'));
        result.put("format", format);
        result.put("instances", entry.data().numInstances());
        result.put("attributes", entry.data().numAttributes());
        result.put("bytes", Files.size(target));
        result.put("sha256", sha256(target));
        return result;
    }

    private static int attributeIndex(Instances data, String name) {
        for (int index = 0; index < data.numAttributes(); index++) {
            if (data.attribute(index).name().equals(name)) {
                return index;
            }
        }
        throw new IllegalArgumentException("Unknown attribute: " + name);
    }

    private static int requiredAttribute(Map<String, Object> arguments, String name, Instances data) {
        int index = ToolSupport.optionalInt(arguments, name, 0);
        if (index < 1 || index > data.numAttributes()) {
            throw new IllegalArgumentException(name + " is a required one-based attribute index");
        }
        return index - 1;
    }

    private static void addValue(ArrayNode row, Instance instance, int index) {
        if (instance.isMissing(index)) {
            row.addNull();
        } else if (instance.attribute(index).isNumeric()) {
            row.add(instance.value(index));
        } else {
            row.add(instance.toString(index));
        }
    }

    private static double mean(Instances data, int index) {
        double total = 0;
        int count = 0;
        for (Instance instance : data) {
            if (!instance.isMissing(index)) {
                total += instance.value(index);
                count++;
            }
        }
        return count == 0 ? Double.NaN : total / count;
    }

    private static double missingPercent(Instances data, int index) {
        int missing = 0;
        for (Instance instance : data) {
            if (instance.isMissing(index)) {
                missing++;
            }
        }
        return percent(missing, data.numInstances());
    }

    private static double percent(double count, double total) {
        return total == 0 ? Double.NaN : count * 100.0 / total;
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = Files.newInputStream(path)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    digest.update(buffer, 0, read);
                }
            }
            return java.util.HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
