package dev.wekamcp.automl;

import com.fasterxml.jackson.annotation.JsonProperty;
import tools.jackson.dataformat.yaml.YAMLMapper;
import weka.core.Utils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class SearchSpace {
    public record SearchCandidate(String classifierClass, String[] options) {
        public SearchCandidate {
            options = options.clone();
        }

        @Override
        public String[] options() {
            return options.clone();
        }
    }

    private record Entry(@JsonProperty("class") String className, List<String> candidates) {
    }

    private record SearchSpaceData(List<Entry> classifiers) {
    }

    private final List<Entry> entries;

    private SearchSpace(List<Entry> entries) {
        this.entries = entries;
    }

    public static SearchSpace load(Path path) throws IOException {
        SearchSpaceData data = new YAMLMapper().readValue(path.toFile(), SearchSpaceData.class);
        return new SearchSpace(data.classifiers() == null ? List.of() : data.classifiers());
    }

    public List<SearchCandidate> candidates(List<String> defaultClasses, String search, int maximum, long seed)
            throws Exception {
        List<SearchCandidate> candidates = new ArrayList<>();
        candidates.add(new SearchCandidate("weka.classifiers.rules.ZeroR", new String[0]));
        for (String className : defaultClasses) {
            Entry entry = entries.stream().filter(item -> item.className().equals(className)).findFirst().orElse(null);
            if (entry == null) {
                continue;
            }
            if ("defaults".equals(search)) {
                candidates.add(new SearchCandidate(className, new String[0]));
            } else {
                for (String options : entry.candidates()) {
                    candidates.add(new SearchCandidate(className, Utils.splitOptions(options)));
                }
            }
        }
        if ("random".equals(search) && candidates.size() > 1) {
            List<SearchCandidate> tail = new ArrayList<>(candidates.subList(1, candidates.size()));
            java.util.Collections.shuffle(tail, new Random(seed));
            candidates = new ArrayList<>(List.of(candidates.get(0)));
            candidates.addAll(tail);
        } else if (!List.of("defaults", "grid", "random", "halving").contains(search)) {
            throw new IllegalArgumentException("search must be defaults, grid, random or halving");
        }
        return List.copyOf(candidates.subList(0, Math.min(maximum, candidates.size())));
    }

    public static List<SearchCandidate> customCandidates(List<?> definitions, int maximum, long seed, String search,
            boolean includeBaseline, int maxOptions, int maxTokenLength) {
        if (!List.of("defaults", "grid", "random", "halving").contains(search)) {
            throw new IllegalArgumentException("search must be defaults, grid, random or halving");
        }
        if (definitions.isEmpty()) {
            throw new IllegalArgumentException("candidates must not be empty");
        }
        List<SearchCandidate> candidates = new ArrayList<>();
        for (Object definition : definitions) {
            if (!(definition instanceof java.util.Map<?, ?> input)) {
                throw new IllegalArgumentException("Each candidate must be an object with classifier_class and optional options");
            }
            Map<String, Object> candidate = new java.util.LinkedHashMap<>();
            for (Map.Entry<?, ?> field : input.entrySet()) {
                if (!(field.getKey() instanceof String key)
                        || !java.util.Set.of("classifier_class", "options").contains(key)) {
                    throw new IllegalArgumentException("Candidate fields must be classifier_class and options only");
                }
                candidate.put((String) field.getKey(), field.getValue());
            }
            String className = dev.wekamcp.tools.ToolSupport.requiredString(candidate, "classifier_class");
            String[] options = dev.wekamcp.tools.ToolSupport.options(candidate, maxOptions, maxTokenLength);
            candidates.add(new SearchCandidate(className, options));
        }
        for (int index = 0; index < candidates.size(); index++) {
            if (candidates.get(index).classifierClass().equals("weka.classifiers.rules.ZeroR")) {
                candidates.add(0, candidates.remove(index));
                break;
            }
        }
        if (includeBaseline && candidates.stream().noneMatch(
                candidate -> candidate.classifierClass().equals("weka.classifiers.rules.ZeroR"))) {
            candidates.add(0, new SearchCandidate("weka.classifiers.rules.ZeroR", new String[0]));
        }
        if ("random".equals(search) && candidates.size() > 1) {
            List<SearchCandidate> tail = new ArrayList<>(candidates.subList(1, candidates.size()));
            java.util.Collections.shuffle(tail, new Random(seed));
            candidates = new ArrayList<>(List.of(candidates.get(0)));
            candidates.addAll(tail);
        }
        return List.copyOf(candidates.subList(0, Math.min(maximum, candidates.size())));
    }
}
