package weka.mcp.gui;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;
import weka.core.Instances;
import weka.gui.explorer.Explorer;
import weka.gui.visualize.VisualizePanel;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

final class ExplorerVisualizationOperations {
    private static final JsonMapper JSON = new JsonMapper();
    private static final int MAX_IMAGE_BYTES = 2_000_000;
    private final Explorer explorer;
    private final PluginWorkspace workspace;
    private JFrame frame;
    private VisualizePanel chart;
    private Instances dataset;
    private String xAttribute;
    private String yAttribute;
    private String colorAttribute;

    ExplorerVisualizationOperations(Explorer explorer, PluginWorkspace workspace) {
        this.explorer = explorer;
        this.workspace = workspace;
    }

    void appendState(ObjectNode result) {
        result.put("visualization_open", frame != null && frame.isDisplayable());
        result.put("visualization_relation", dataset == null ? "" : dataset.relationName());
        result.put("visualization_x_attribute", xAttribute == null ? "" : xAttribute);
        result.put("visualization_y_attribute", yAttribute == null ? "" : yAttribute);
        result.put("visualization_color_attribute", colorAttribute == null ? "" : colorAttribute);
    }

    ObjectNode show(JsonNode arguments) throws Exception {
        close();
        return ExplorerSession.onEdt(() -> {
            Instances data = explorer.getPreprocessPanel().getInstances();
            if (data == null || data.numInstances() == 0) {
                throw new IllegalStateException("Load a dataset in Explorer Preprocess first");
            }
            if (data.numAttributes() < 2) {
                throw new IllegalArgumentException("A scatter plot requires at least two attributes");
            }
            xAttribute = attribute(data, arguments, "x_attribute", data.attribute(0).name(), false);
            yAttribute = attribute(data, arguments, "y_attribute", data.attribute(1).name(), false);
            colorAttribute = attribute(data, arguments, "color_attribute",
                    data.classIndex() < 0 ? data.attribute(data.numAttributes() - 1).name()
                            : data.classAttribute().name(), true);
            int width = dimension(arguments, "width", 1200, 320, 1600);
            int height = dimension(arguments, "height", 800, 240, 1200);
            VisualizePanel created = new VisualizePanel();
            created.setInstances(data);
            created.setXIndex(data.attribute(xAttribute).index());
            created.setYIndex(data.attribute(yAttribute).index());
            created.setColourIndex(colorAttribute.isEmpty() ? -1 : data.attribute(colorAttribute).index());
            JFrame createdFrame = new JFrame("Weka MCP - " + data.relationName());
            createdFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            createdFrame.setLayout(new BorderLayout());
            createdFrame.add(created, BorderLayout.CENTER);
            createdFrame.setSize(width, height);
            createdFrame.setLocationRelativeTo(explorer);
            createdFrame.setVisible(true);
            frame = createdFrame;
            chart = created;
            dataset = data;
            explorer.getPanels().stream().filter(weka.gui.explorer.VisualizePanel.class::isInstance)
                    .findFirst().ifPresent(panel -> explorer.getTabbedPane().setSelectedComponent((java.awt.Component) panel));
            ObjectNode result = JSON.createObjectNode();
            result.put("displayed", true);
            result.put("target", "dataset_scatter");
            result.put("relation_name", data.relationName());
            result.put("x_attribute", xAttribute);
            result.put("y_attribute", yAttribute);
            result.put("color_attribute", colorAttribute);
            result.put("width", width);
            result.put("height", height);
            result.put("window", "Weka native scatter plot");
            result.put("visualization_open", true);
            return result;
        });
    }

    Capture capture(JsonNode arguments) throws Exception {
        if (chart == null) {
            throw new IllegalStateException("Call weka_explorer_visualize before capturing an image");
        }
        if (workspace == null) {
            throw new IllegalStateException("Image export is not configured");
        }
        int width = dimension(arguments, "width", 1200, 320, 1600);
        int height = dimension(arguments, "height", 800, 240, 1200);
        BufferedImage image = ExplorerSession.onEdt(() -> {
            chart.setSize(width, height);
            chart.validate();
            BufferedImage rendered = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D graphics = rendered.createGraphics();
            try {
                chart.paint(graphics);
            } finally {
                graphics.dispose();
            }
            return rendered;
        });
        Path directory = workspace.workspaceDirectory().resolve("data/exports/visualizations");
        Files.createDirectories(directory);
        Path temporary = Files.createTempFile(directory, ".plot-", ".png");
        Path destination = directory.resolve(UUID.randomUUID() + ".png");
        try {
            ImageIO.write(image, "png", temporary.toFile());
            if (Files.size(temporary) > MAX_IMAGE_BYTES) {
                throw new IllegalStateException("PNG exceeds the 2000000-byte response limit");
            }
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(temporary);
        }
        byte[] png = Files.readAllBytes(destination);
        ObjectNode metadata = JSON.createObjectNode();
        metadata.put("format", "png");
        metadata.put("source", "native_visualization");
        metadata.put("target", "dataset_scatter");
        metadata.put("relation_name", dataset.relationName());
        metadata.put("x_attribute", xAttribute);
        metadata.put("y_attribute", yAttribute);
        metadata.put("color_attribute", colorAttribute);
        metadata.put("width", width);
        metadata.put("height", height);
        metadata.put("bytes", png.length);
        metadata.put("image_path", destination.toAbsolutePath().toString());
        metadata.put("relative_path", "exports/visualizations/" + destination.getFileName());
        return new Capture(metadata, png, "image/png");
    }

    ObjectNode closeTool() throws Exception {
        return ExplorerSession.onEdt(() -> {
            boolean closed = frame != null;
            if (frame != null) {
                frame.dispose();
            }
            frame = null;
            chart = null;
            dataset = null;
            xAttribute = null;
            yAttribute = null;
            colorAttribute = null;
            ObjectNode result = JSON.createObjectNode();
            result.put("closed", closed);
            result.put("visualization_open", false);
            return result;
        });
    }

    void close() {
        try {
            closeTool();
        } catch (Exception ignored) {
        }
    }

    private static String attribute(Instances data, JsonNode arguments, String name, String defaultValue,
            boolean allowEmpty) {
        if (!arguments.has(name)) {
            return defaultValue;
        }
        if (!arguments.get(name).isTextual()) {
            throw new IllegalArgumentException(name + " must be a string");
        }
        String value = arguments.get(name).asText();
        if (value.isEmpty() && allowEmpty) {
            return "";
        }
        if (data.attribute(value) == null) {
            throw new IllegalArgumentException("Unknown attribute: " + value);
        }
        return value;
    }

    private static int dimension(JsonNode arguments, String name, int defaultValue, int minimum, int maximum) {
        if (arguments.has(name) && !arguments.get(name).isInt()) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
        int value = arguments.has(name) ? arguments.get(name).intValue() : defaultValue;
        if (value < minimum || value > maximum) {
            throw new IllegalArgumentException(name + " must be between " + minimum + " and " + maximum);
        }
        return value;
    }

    static ObjectNode visualizeProperties() {
        ObjectNode properties = JSON.createObjectNode();
        properties.withObject("x_attribute").put("type", "string").put("description", "Attribute for the X axis");
        properties.withObject("y_attribute").put("type", "string").put("description", "Attribute for the Y axis");
        properties.withObject("color_attribute").put("type", "string")
                .put("description", "Optional colour attribute; an empty string disables colouring");
        addDimensions(properties);
        return properties;
    }

    static ObjectNode captureProperties() {
        ObjectNode properties = JSON.createObjectNode();
        addDimensions(properties);
        return properties;
    }

    private static void addDimensions(ObjectNode properties) {
        properties.withObject("width").put("type", "integer")
                .put("description", "Image width, 320 to 1600; defaults to 1200");
        properties.withObject("height").put("type", "integer")
                .put("description", "Image height, 240 to 1200; defaults to 800");
    }

    record Capture(ObjectNode metadata, byte[] png, String mimeType) {
    }
}
