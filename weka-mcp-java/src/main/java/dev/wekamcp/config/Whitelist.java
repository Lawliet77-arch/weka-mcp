package dev.wekamcp.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import tools.jackson.dataformat.yaml.YAMLMapper;
import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.OptionHandler;
import weka.filters.Filter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class Whitelist {
    public enum Kind {
        CLASSIFIER(Classifier.class),
        FILTER(Filter.class),
        CLUSTERER(Clusterer.class),
        ASSOCIATOR(weka.associations.Associator.class);

        private final Class<?> expectedType;

        Kind(Class<?> expectedType) {
            this.expectedType = expectedType;
        }
    }

    public record OptionRule(int args, String type, Double min, Double max) {
        public OptionRule {
            if (args < 0) {
                throw new IllegalArgumentException("Option args must be zero or greater");
            }
            type = type == null ? "string" : type;
            if (!Set.of("string", "number", "integer").contains(type)) {
                throw new IllegalArgumentException("Unsupported option type: " + type);
            }
        }
    }

    public record AlgorithmSpec(
            @JsonProperty("class") String className,
            @JsonProperty("allowed_options") Map<String, OptionRule> allowedOptions) {
        public AlgorithmSpec {
            if (className == null || className.isBlank()) {
                throw new IllegalArgumentException("Whitelist class must not be blank");
            }
            allowedOptions = allowedOptions == null ? Map.of() : Map.copyOf(allowedOptions);
        }
    }

    private record WhitelistData(
            List<AlgorithmSpec> classifiers,
            List<AlgorithmSpec> filters,
            List<AlgorithmSpec> clusterers,
            List<AlgorithmSpec> associators) {
    }

    private final Map<Kind, Map<String, AlgorithmSpec>> specifications;

    private Whitelist(Map<Kind, Map<String, AlgorithmSpec>> specifications) {
        this.specifications = specifications;
    }

    public static Whitelist load(Path path) throws IOException {
        WhitelistData data = new YAMLMapper().readValue(path.toFile(), WhitelistData.class);
        Map<Kind, Map<String, AlgorithmSpec>> specifications = new EnumMap<>(Kind.class);
        specifications.put(Kind.CLASSIFIER, index(data.classifiers()));
        specifications.put(Kind.FILTER, index(data.filters()));
        specifications.put(Kind.CLUSTERER, index(data.clusterers()));
        specifications.put(Kind.ASSOCIATOR, index(data.associators()));
        return new Whitelist(specifications);
    }

    public void selfCheck() {
        List<String> failures = new ArrayList<>();
        for (Map.Entry<Kind, Map<String, AlgorithmSpec>> group : specifications.entrySet()) {
            for (AlgorithmSpec specification : group.getValue().values()) {
                try {
                    instantiate(group.getKey(), specification);
                } catch (ReflectiveOperationException | RuntimeException e) {
                    failures.add(specification.className() + ": " + e.getMessage());
                }
            }
        }
        if (!failures.isEmpty()) {
            throw new IllegalStateException("Whitelist self-check failed: " + String.join("; ", failures));
        }
    }

    public Object create(Kind kind, String className, String[] options) {
        AlgorithmSpec specification = get(kind, className);
        validateOptions(specification, options == null ? new String[0] : options);
        try {
            Object instance = instantiate(kind, specification);
            if (instance instanceof OptionHandler handler) {
                handler.setOptions(options == null ? new String[0] : options.clone());
            }
            return instance;
        } catch (Exception e) {
            throw new IllegalArgumentException("Cannot apply options to whitelisted class " + className + ": " + e.getMessage(), e);
        }
    }

    public AlgorithmSpec get(Kind kind, String className) {
        AlgorithmSpec specification = specifications.get(kind).get(className);
        if (specification == null) {
            throw new IllegalArgumentException(className + " is not in the " + kind.name().toLowerCase() + " whitelist");
        }
        return specification;
    }

    public List<AlgorithmSpec> list(Kind kind) {
        return List.copyOf(specifications.get(kind).values());
    }

    private static Map<String, AlgorithmSpec> index(List<AlgorithmSpec> entries) {
        Map<String, AlgorithmSpec> result = new LinkedHashMap<>();
        if (entries != null) {
            for (AlgorithmSpec entry : entries) {
                if (result.putIfAbsent(entry.className(), entry) != null) {
                    throw new IllegalArgumentException("Duplicate whitelist class: " + entry.className());
                }
            }
        }
        return result;
    }

    private static Object instantiate(Kind kind, AlgorithmSpec specification) throws ReflectiveOperationException {
        Object instance = Class.forName(specification.className()).getDeclaredConstructor().newInstance();
        if (!kind.expectedType.isInstance(instance)) {
            throw new IllegalArgumentException(specification.className() + " has an incompatible Weka type");
        }
        if (instance instanceof OptionHandler handler) {
            handler.getOptions();
        }
        return instance;
    }

    private static void validateOptions(AlgorithmSpec specification, String[] options) {
        Set<String> seen = new HashSet<>();
        int index = 0;
        while (index < options.length) {
            String flag = options[index];
            OptionRule rule = specification.allowedOptions().get(flag);
            if (rule == null) {
                throw new IllegalArgumentException("Option " + flag + " is not allowed for " + specification.className());
            }
            if (!seen.add(flag)) {
                throw new IllegalArgumentException("Option " + flag + " must not be repeated");
            }
            if (index + rule.args() >= options.length) {
                throw new IllegalArgumentException("Option " + flag + " expects " + rule.args() + " value(s)");
            }
            for (int offset = 1; offset <= rule.args(); offset++) {
                validateValue(flag, options[index + offset], rule);
            }
            index += rule.args() + 1;
        }
    }

    private static void validateValue(String flag, String value, OptionRule rule) {
        if (!rule.type().equals("string")) {
            final double numeric;
            try {
                numeric = rule.type().equals("integer") ? Long.parseLong(value) : Double.parseDouble(value);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Option " + flag + " expects a " + rule.type() + " value");
            }
            if (!Double.isFinite(numeric)
                    || (rule.min() != null && numeric < rule.min())
                    || (rule.max() != null && numeric > rule.max())) {
                throw new IllegalArgumentException("Option " + flag + " value is outside its allowed range");
            }
        }
    }
}
