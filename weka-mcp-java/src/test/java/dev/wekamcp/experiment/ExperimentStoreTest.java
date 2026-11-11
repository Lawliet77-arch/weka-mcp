package dev.wekamcp.experiment;

import dev.wekamcp.config.ServerConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExperimentStoreTest {
    @TempDir
    Path directory;

    @Test
    void savesLoadsComparesAndDeletesExperiments() throws Exception {
        Path configPath = directory.resolve("config.properties");
        Files.writeString(configPath, "dataRoot=data\nexperimentRoot=experiments\n");
        ServerConfig config = ServerConfig.load(configPath);
        ExperimentStore store = new ExperimentStore(config);
        Instances data = new Instances(new StringReader("""
                @relation tiny
                @attribute x numeric
                @attribute class {a,b}
                @data
                1,a
                2,a
                3,b
                4,b
                """));
        data.setClassIndex(1);
        J48 classifier = new J48();
        classifier.buildClassifier(data);
        ExperimentRecord record = new ExperimentRecord();
        record.classifierClass = J48.class.getName();
        record.options = new String[0];
        record.protocol = "holdout";
        record.classIndex = 1;
        record.instances = data.numInstances();
        record.attributes = data.numAttributes();

        ExperimentRecord saved = store.create(record, data, classifier, data);
        assertNotNull(saved.id);
        assertNotNull(saved.dataHash);
        assertEquals(1, store.count());
        assertEquals(saved.id, store.get(saved.id).id);
        assertEquals(1, store.list(0, 10).size());

        ModelCodec.ModelPayload loaded = store.loadModel(saved.id);
        assertEquals(classifier.classifyInstance(data.instance(0)),
                loaded.classifier().classifyInstance(data.instance(0)));
        assertEquals(2, store.loadDataset(saved.id).numAttributes());

        store.delete(saved.id);
        assertEquals(0, store.count());
        assertThrows(IllegalArgumentException.class, () -> store.get(saved.id));
    }

    @Test
    void rejectsUnexpectedSerializedClasses() throws Exception {
        Path configPath = directory.resolve("config.properties");
        Files.writeString(configPath, "dataRoot=data\nexperimentRoot=experiments\n");
        ServerConfig config = ServerConfig.load(configPath);
        ExperimentStore store = new ExperimentStore(config);
        Path model = config.experimentRoot().resolve("unexpected.model");
        try (ObjectOutputStream output = new ObjectOutputStream(Files.newOutputStream(model))) {
            output.writeObject(new UnexpectedPayload());
        }
        assertThrows(java.io.InvalidClassException.class,
                () -> ModelCodec.read(model, 1024, J48.class.getName(), false));
    }

    @Test
    void readsClassifierPayloadSerializedWithTheOriginalLayout() throws Exception {
        ModelCodec.ModelPayload payload = ModelCodec.read(
                Path.of("src/test/resources/legacy-classifier-1.4.0.model"), 1_000_000, J48.class.getName(), false);
        Instances data = dev.wekamcp.engine.DatasetIo.load(Path.of("src/test/resources/iris.arff"), "arff");
        data.setClassIndex(data.numAttributes() - 1);
        J48 expected = new J48();
        expected.buildClassifier(data);
        for (Instance instance : data) {
            assertEquals(expected.classifyInstance(instance), payload.classifier().classifyInstance(instance));
        }
    }

    private static final class UnexpectedPayload implements Serializable {
        private static final long serialVersionUID = 1L;
    }
}
