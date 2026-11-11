package dev.wekamcp.experiment;

import dev.wekamcp.engine.Preprocessing.FilterSpec;
import tools.jackson.databind.JsonNode;

import java.util.List;

public final class ExperimentRecord {
    public int schemaVersion = 1;
    public String id;
    public String status;
    public String createdAt;
    public String javaVersion;
    public String wekaVersion;
    public String parentExperimentId;
    public String kind = "CLASSIFIER";
    public String classifierClass;
    public String clustererClass;
    public String sourcePath;
    public String datasetRole = "TRAINING";
    public boolean retrainable = true;
    public JsonNode interoperability;
    public int removedClassIndex = -1;
    public int[] assignments;
    public String[] options = new String[0];
    public List<FilterSpec> preprocessing = List.of();
    public int classIndex = -1;
    public int instances;
    public int attributes;
    public String dataHash;
    public String headerHash;
    public long seed;
    public String protocol;
    public String optimize;
    public String selectionMetric;
    public double validationScore = Double.NaN;
    public double testScore = Double.NaN;
    public JsonNode validationMetrics;
    public JsonNode testMetrics;
    public JsonNode candidates;
    public JsonNode search;
    public JsonNode stream;
    public boolean finalModelTrainedOnFullData = true;
}
