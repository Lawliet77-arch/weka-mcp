package dev.wekamcp.experiment;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.clusterers.Clusterer;
import weka.core.Instances;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;

public final class ModelCodec {
    public record ModelPayload(Classifier classifier, Clusterer clusterer, Instances trainingHeader)
            implements Serializable {
        private static final long serialVersionUID = 1L;

        public ModelPayload {
            if ((classifier == null) == (clusterer == null)) {
                throw new IllegalArgumentException("Payload must contain exactly one classifier or clusterer");
            }
            trainingHeader = new Instances(trainingHeader, 0);
        }

        public ModelPayload(Object model, Instances trainingHeader) {
            this(model instanceof Classifier value ? value : null,
                    model instanceof Clusterer value ? value : null, trainingHeader);
        }

        public Classifier classifier() {
            if (classifier == null) {
                throw new IllegalArgumentException("Experiment does not contain a classifier");
            }
            return classifier;
        }

        public Clusterer clusterer() {
            if (clusterer == null) {
                throw new IllegalArgumentException("Experiment does not contain a clusterer");
            }
            return clusterer;
        }
    }

    public record StandardModelPayload(Object model, Instances trainingHeader, boolean headerIncluded) {
    }

    public static StandardModelPayload readStandard(Path path, long maxBytes,
            java.util.function.Predicate<String> allowedWekaType) throws IOException, ClassNotFoundException {
        long size = Files.size(path);
        if (size > maxBytes) {
            throw new IllegalArgumentException("Model exceeds maxModelBytes=" + maxBytes);
        }
        java.util.concurrent.atomic.AtomicReference<String> rejected = new java.util.concurrent.atomic.AtomicReference<>();
        ObjectInputFilter filter = info -> {
            if (info.depth() > 100 || info.references() > 1_000_000) {
                return ObjectInputFilter.Status.REJECTED;
            }
            Class<?> type = info.serialClass();
            if (type == null) {
                return ObjectInputFilter.Status.UNDECIDED;
            }
            String name = type.getName();
            boolean allowed = name.startsWith("[") || allowedWekaType.test(name) || safeJavaType(name);
            if (!allowed) {
                rejected.compareAndSet(null, name);
            }
            return allowed ? ObjectInputFilter.Status.ALLOWED : ObjectInputFilter.Status.REJECTED;
        };
        try (ObjectInputStream input = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            input.setObjectInputFilter(filter);
            Object model = input.readObject();
            if (!(model instanceof Classifier) && !(model instanceof Clusterer)) {
                throw new IllegalArgumentException("Model must be a Weka classifier or clusterer");
            }
            Instances header = null;
            try {
                Object value = input.readObject();
                if (!(value instanceof Instances)) {
                    throw new IllegalArgumentException("The second object in a Weka model bundle must be a training header");
                }
                header = new Instances((Instances) value, 0);
            } catch (java.io.EOFException ignored) {
            }
            return new StandardModelPayload(model, header, header != null);
        } catch (IOException | ClassNotFoundException e) {
            if (rejected.get() != null) {
                throw new IOException("Serialized type is not allowed: " + rejected.get(), e);
            }
            throw e;
        }
    }

    private static boolean safeJavaType(String name) {
        if (name.startsWith("java.util.") || name.startsWith("java.math.") || name.startsWith("java.time.")) {
            return true;
        }
        return name.equals("java.lang.Object") || name.equals("java.lang.Number")
                || name.equals("java.lang.String") || name.equals("java.lang.StringBuilder")
                || name.equals("java.lang.StringBuffer") || name.equals("java.lang.Enum")
                || name.equals("java.lang.Boolean") || name.equals("java.lang.Byte")
                || name.equals("java.lang.Character") || name.equals("java.lang.Short")
                || name.equals("java.lang.Integer") || name.equals("java.lang.Long")
                || name.equals("java.lang.Float") || name.equals("java.lang.Double");
    }

    private ModelCodec() {
    }

    public static void write(Path path, Object model, Instances trainingHeader) throws IOException {
        try (ObjectOutputStream output = new ObjectOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            output.writeObject(new ModelPayload(model, trainingHeader));
        }
    }

    public static ModelPayload read(Path path, long maxBytes, String expectedClassifierClass, boolean filtered)
            throws IOException, ClassNotFoundException {
        return read(path, maxBytes, expectedClassifierClass, filtered, false);
    }

    public static ModelPayload read(Path path, long maxBytes, String expectedClass, boolean filtered, boolean clusterer)
            throws IOException, ClassNotFoundException {
        long size = Files.size(path);
        if (size > maxBytes) {
            throw new IllegalArgumentException("Model exceeds maxModelBytes=" + maxBytes);
        }
        ObjectInputFilter filter = info -> {
            if (info.depth() > 100 || info.references() > 1_000_000) {
                return ObjectInputFilter.Status.REJECTED;
            }
            Class<?> type = info.serialClass();
            if (type == null) {
                return ObjectInputFilter.Status.UNDECIDED;
            }
            String name = type.getName();
            if (name.equals(ModelPayload.class.getName()) || name.startsWith("weka.")
                    || name.startsWith("java.") || name.startsWith("[")) {
                return ObjectInputFilter.Status.ALLOWED;
            }
            return ObjectInputFilter.Status.REJECTED;
        };
        try (ObjectInputStream input = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            input.setObjectInputFilter(filter);
            Object value = input.readObject();
            if (!(value instanceof ModelPayload payload)) {
                throw new IllegalArgumentException("Unexpected model payload type");
            }
            Object model = clusterer ? payload.clusterer() : payload.classifier();
            if (filtered) {
                if (!(model instanceof FilteredClassifier)) {
                    throw new IllegalArgumentException("Expected a filtered classifier");
                }
            } else if (!model.getClass().getName().equals(expectedClass)) {
                throw new IllegalArgumentException("Model class does not match the experiment manifest");
            }
            return payload;
        }
    }
}
