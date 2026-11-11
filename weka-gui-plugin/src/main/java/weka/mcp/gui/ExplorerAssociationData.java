package weka.mcp.gui;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

final class ExplorerAssociationData {
    private static final int MAX_EXPANDED_ATTRIBUTES = 2000;

    private ExplorerAssociationData() {
    }

    static EncodedData forFpGrowth(Instances input) {
        List<Attribute> attributes = new ArrayList<>();
        List<int[]> sources = new ArrayList<>();
        boolean expanded = false;
        for (int index = 0; index < input.numAttributes(); index++) {
            Attribute attribute = input.attribute(index);
            if (!attribute.isNominal()) {
                throw new IllegalArgumentException("Association algorithms require nominal attributes; "
                        + "apply Discretize before association mining");
            }
            if (attribute.numValues() <= 2) {
                attributes.add(new Attribute(attribute.name(), values(attribute)));
                sources.add(new int[]{index, -1});
                continue;
            }
            expanded = true;
            for (int value = 0; value < attribute.numValues(); value++) {
                if (attributes.size() >= MAX_EXPANDED_ATTRIBUTES) {
                    throw new IllegalArgumentException("Binary nominal expansion exceeds "
                            + MAX_EXPANDED_ATTRIBUTES + " attributes");
                }
                attributes.add(new Attribute(attribute.name() + "=" + attribute.value(value), List.of("0", "1")));
                sources.add(new int[]{index, value});
            }
        }
        Instances output = new Instances(input.relationName(), new ArrayList<>(attributes), input.numInstances());
        for (int row = 0; row < input.numInstances(); row++) {
            Instance original = input.instance(row);
            DenseInstance encoded = new DenseInstance(sources.size());
            encoded.setDataset(output);
            for (int column = 0; column < sources.size(); column++) {
                int[] source = sources.get(column);
                if (original.isMissing(source[0])) {
                    encoded.setMissing(column);
                } else if (source[1] < 0) {
                    encoded.setValue(column, original.value(source[0]));
                } else {
                    encoded.setValue(column, original.value(source[0]) == source[1] ? 1 : 0);
                }
            }
            output.add(encoded);
        }
        output.setClassIndex(-1);
        return new EncodedData(output, expanded);
    }

    private static List<String> values(Attribute attribute) {
        List<String> values = new ArrayList<>();
        for (int index = 0; index < attribute.numValues(); index++) {
            values.add(attribute.value(index));
        }
        return values;
    }

    record EncodedData(Instances instances, boolean expanded) {
    }
}
