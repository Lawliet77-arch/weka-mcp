package dev.wekamcp.engine;

import dev.wekamcp.config.Whitelist;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Preprocessing {
    public record FilterSpec(String filterClass, String[] options) {
        public FilterSpec {
            filterClass = java.util.Objects.requireNonNull(filterClass);
            options = options.clone();
        }

        @Override
        public String[] options() {
            return options.clone();
        }
    }

    private static final String REPLACE_MISSING = "weka.filters.unsupervised.attribute.ReplaceMissingValues";
    private static final String STANDARDIZE = "weka.filters.unsupervised.attribute.Standardize";
    private static final Set<String> SCALE_SENSITIVE = Set.of(
            "weka.classifiers.functions.Logistic",
            "weka.classifiers.functions.SMO",
            "weka.classifiers.functions.SMOreg",
            "weka.classifiers.functions.LinearRegression",
            "weka.classifiers.lazy.IBk");

    private Preprocessing() {
    }

    public static List<FilterSpec> automatic(Instances data, String classifierClass) {
        List<FilterSpec> filters = new ArrayList<>();
        if (hasStringPredictor(data)) {
            filters.add(new FilterSpec("weka.filters.unsupervised.attribute.StringToWordVector", new String[0]));
        }
        filters.add(new FilterSpec(REPLACE_MISSING, new String[0]));
        if (SCALE_SENSITIVE.contains(classifierClass) && hasNumericPredictor(data)) {
            filters.add(new FilterSpec(STANDARDIZE, new String[0]));
        }
        return List.copyOf(filters);
    }

    public static Classifier wrap(Classifier base, List<FilterSpec> specifications, Whitelist whitelist) {
        if (specifications.isEmpty()) {
            return base;
        }
        Filter[] filters = new Filter[specifications.size()];
        for (int index = 0; index < specifications.size(); index++) {
            FilterSpec specification = specifications.get(index);
            filters[index] = (Filter) whitelist.create(
                    Whitelist.Kind.FILTER, specification.filterClass(), specification.options());
        }
        Filter pipeline;
        if (filters.length == 1) {
            pipeline = filters[0];
        } else {
            MultiFilter multiFilter = new MultiFilter();
            multiFilter.setFilters(filters);
            pipeline = multiFilter;
        }
        FilteredClassifier classifier = new FilteredClassifier();
        classifier.setFilter(pipeline);
        classifier.setClassifier(base);
        return classifier;
    }

    private static boolean hasStringPredictor(Instances data) {
        for (int index = 0; index < data.numAttributes(); index++) {
            if (index != data.classIndex() && data.attribute(index).isString()) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasNumericPredictor(Instances data) {
        for (int index = 0; index < data.numAttributes(); index++) {
            if (index != data.classIndex() && data.attribute(index).isNumeric()) {
                return true;
            }
        }
        return false;
    }
}
