package dev.wekamcp.engine;

import weka.core.Instances;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.Loader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;

public final class DatasetIo {
    private DatasetIo() {
    }

    public static Instances load(Path path, String format) throws Exception {
        try (LoaderResource opened = open(path, format)) {
            return opened.loader().getDataSet();
        }
    }

    public static LoaderResource open(Path path, String format) throws IOException {
        if (!Set.of("auto", "arff", "csv").contains(format)) {
            throw new IllegalArgumentException("format must be auto, arff or csv");
        }
        String filename = path.getFileName().toString().toLowerCase(Locale.ROOT);
        boolean arff = "arff".equals(format) || "auto".equals(format) && filename.endsWith(".arff");
        if ("auto".equals(format) && !arff && !filename.endsWith(".csv")) {
            throw new IllegalArgumentException("auto format requires an .arff or .csv filename");
        }
        AbstractFileLoader loader = arff ? new ArffLoader() : new CSVLoader();
        InputStream input = Files.newInputStream(path);
        try {
            loader.setSource(input);
            return new LoaderResource(loader, input);
        } catch (IOException | RuntimeException | Error failure) {
            try {
                input.close();
            } catch (IOException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    public static final class LoaderResource implements AutoCloseable {
        private final Loader loader;
        private final InputStream input;

        private LoaderResource(Loader loader, InputStream input) {
            this.loader = loader;
            this.input = input;
        }

        public Loader loader() {
            return loader;
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}
