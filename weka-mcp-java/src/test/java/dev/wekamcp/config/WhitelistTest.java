package dev.wekamcp.config;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import weka.classifiers.trees.J48;
import weka.core.WekaPackageManager;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhitelistTest {
    private static Whitelist whitelist;

    @BeforeAll
    static void load() throws Exception {
        WekaPackageManager.loadPackages(false, true, false);
        whitelist = Whitelist.load(Path.of("weka-whitelist.yaml"));
        whitelist.selfCheck();
    }

    @Test
    void loadsAndInstantiatesEveryWhitelistedClass() {
        assertEquals(36, whitelist.list(Whitelist.Kind.CLASSIFIER).size());
        assertEquals(18, whitelist.list(Whitelist.Kind.FILTER).size());
        assertEquals(6, whitelist.list(Whitelist.Kind.CLUSTERER).size());
        assertEquals(2, whitelist.list(Whitelist.Kind.ASSOCIATOR).size());
        J48 classifier = (J48) whitelist.create(
                Whitelist.Kind.CLASSIFIER, "weka.classifiers.trees.J48", new String[]{"-C", "0.25", "-M", "2"});
        assertEquals(0.25f, classifier.getConfidenceFactor());
        assertEquals(2, classifier.getMinNumObj());
    }

    @Test
    void rejectsUnknownClassesOptionsAndInvalidValues() {
        assertThrows(IllegalArgumentException.class, () -> whitelist.create(
                Whitelist.Kind.CLASSIFIER, "weka.classifiers.functions.DL4J", new String[0]));
        assertThrows(IllegalArgumentException.class, () -> whitelist.create(
                Whitelist.Kind.CLASSIFIER, "weka.classifiers.trees.J48", new String[]{"-X"}));
        assertThrows(IllegalArgumentException.class, () -> whitelist.create(
                Whitelist.Kind.CLASSIFIER, "weka.classifiers.trees.J48", new String[]{"-C", "2"}));
        assertThrows(IllegalArgumentException.class, () -> whitelist.create(
                Whitelist.Kind.CLASSIFIER, "weka.classifiers.trees.J48", new String[]{"-C"}));
    }

    @Test
    void acceptsNegativeNumbersAsOptionValues() {
        Object clusterer = whitelist.create(
                Whitelist.Kind.CLUSTERER, "weka.clusterers.EM", new String[]{"-N", "-1"});
        assertTrue(clusterer instanceof weka.clusterers.EM);
    }
}
