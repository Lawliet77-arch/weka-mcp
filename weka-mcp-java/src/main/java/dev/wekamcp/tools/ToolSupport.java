package dev.wekamcp.tools;

import io.modelcontextprotocol.spec.McpSchema;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ToolSupport {
    public static final JsonMapper JSON = new JsonMapper();

    private ToolSupport() {
    }

    public static ObjectNode object() {
        return JSON.createObjectNode();
    }

    public static ArrayNode array() {
        return JSON.createArrayNode();
    }

    public static String requiredString(Map<String, Object> arguments, String name) {
        String value = optionalString(arguments, name, null);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " is required");
        }
        return value;
    }

    public static String optionalString(Map<String, Object> arguments, String name, String defaultValue) {
        Object value = arguments.get(name);
        if (value == null) {
            return defaultValue;
        }
        if (!(value instanceof String string)) {
            throw new IllegalArgumentException(name + " must be a string");
        }
        return string;
    }

    public static int optionalInt(Map<String, Object> arguments, String name, int defaultValue) {
        Object value = arguments.get(name);
        if (value == null) {
            return defaultValue;
        }
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        long longValue = number.longValue();
        if (number.doubleValue() != longValue || longValue < Integer.MIN_VALUE || longValue > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        return (int) longValue;
    }

    public static long optionalLong(Map<String, Object> arguments, String name, long defaultValue) {
        Object value = arguments.get(name);
        if (value == null) {
            return defaultValue;
        }
        if (!(value instanceof Number number) || number.doubleValue() != number.longValue()) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        return number.longValue();
    }

    public static double optionalDouble(Map<String, Object> arguments, String name, double defaultValue) {
        Object value = arguments.get(name);
        if (value == null) {
            return defaultValue;
        }
        if (!(value instanceof Number number) || !Double.isFinite(number.doubleValue())) {
            throw new IllegalArgumentException(name + " must be a finite number");
        }
        return number.doubleValue();
    }

    public static boolean optionalBoolean(Map<String, Object> arguments, String name, boolean defaultValue) {
        Object value = arguments.get(name);
        if (value == null) {
            return defaultValue;
        }
        if (!(value instanceof Boolean bool)) {
            throw new IllegalArgumentException(name + " must be a boolean");
        }
        return bool;
    }

    public static String[] options(Map<String, Object> arguments, int maximumLength, int maximumTokenLength) {
        Object value = arguments.get("options");
        if (value == null) {
            return new String[0];
        }
        if (!(value instanceof List<?> list)) {
            throw new IllegalArgumentException("options must be an array of strings");
        }
        if (list.size() > maximumLength) {
            throw new IllegalArgumentException("options exceeds maxOptionsLength=" + maximumLength);
        }
        String[] options = new String[list.size()];
        for (int index = 0; index < list.size(); index++) {
            if (!(list.get(index) instanceof String token)) {
                throw new IllegalArgumentException("options must contain only strings");
            }
            if (token.length() > maximumTokenLength) {
                throw new IllegalArgumentException("option token exceeds maxOptionTokenLength=" + maximumTokenLength);
            }
            options[index] = token;
        }
        return options;
    }

    public static Map<String, Object> schema(Map<String, Object> properties, String... required) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        schema.put("required", List.of(required));
        schema.put("additionalProperties", false);
        return schema;
    }

    public static Map<String, Object> string(String description) {
        return Map.of("type", "string", "description", description);
    }

    public static Map<String, Object> integer(String description, int minimum, int maximum) {
        return Map.of("type", "integer", "description", description, "minimum", minimum, "maximum", maximum);
    }

    public static Map<String, Object> bool(String description) {
        return Map.of("type", "boolean", "description", description);
    }

    public static Map<String, Object> optionsSchema() {
        return Map.of(
                "type", "array",
                "description", "Weka option flags and values; every flag must be allowed by the whitelist",
                "items", Map.of("type", "string"),
                "maxItems", 20);
    }

    public static CallToolResult success(String json) {
        return CallToolResult.builder()
                .content(List.of(new McpSchema.TextContent(json)))
                .isError(false)
                .build();
    }

    public static CallToolResult error(String message) {
        ObjectNode payload = object();
        payload.put("error", message);
        return CallToolResult.builder()
                .content(List.of(new McpSchema.TextContent(payload.toString())))
                .isError(true)
                .build();
    }

    public static void putNumber(ObjectNode target, String name, double value) {
        if (Double.isFinite(value)) {
            target.put(name, value);
        } else {
            target.putNull(name);
        }
    }

    public static List<String> classLabels(weka.core.Instances header) {
        List<String> labels = new ArrayList<>();
        if (header.classIndex() >= 0 && header.classAttribute().isNominal()) {
            for (int index = 0; index < header.classAttribute().numValues(); index++) {
                labels.add(header.classAttribute().value(index));
            }
        }
        return labels;
    }
}
