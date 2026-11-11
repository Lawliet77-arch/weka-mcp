package weka.mcp.gui;

import org.junit.jupiter.api.Test;
import weka.core.WekaPackageManager;
import weka.gui.explorer.Explorer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.zip.ZipFile;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PackageInstallationIT {
    @Test
    void archiveInstallsIntoIsolatedWekaHomeAndRegistersExplorerTab() throws Exception {
        Path archive = Path.of("target/weka-mcp-gui-0.8.0.zip");
        assertTrue(Files.isRegularFile(archive), "GUI package was not assembled: " + archive);
        try (ZipFile zip = new ZipFile(archive.toFile())) {
            assertNotNull(zip.getEntry("Description.props"));
            assertNotNull(zip.getEntry("Explorer.props"));
            assertNotNull(zip.getEntry("weka-mcp-gui-0.8.0.jar"));
            assertNotNull(zip.getEntry("tools/weka-mcp-java-1.6.0.jar"));
            assertNotNull(zip.getEntry("lib/jackson-databind-3.1.4.jar"));
            assertNotNull(zip.getEntry("config/weka-whitelist.yaml"));
            assertNotNull(zip.getEntry("bin/bridge-stdio.sh"));
            assertNotNull(zip.getEntry("bin/bridge-stdio.bat"));
            assertFalse(zip.stream().anyMatch(entry -> entry.getName().startsWith("weka-mcp-gui/")));
        }
        String launcher = Files.readString(Path.of("target/package/bin/bridge-stdio.sh"));
        assertFalse(launcher.contains("@project.version@"));
        assertTrue(launcher.contains("weka-mcp-gui-0.8.0.jar"));
        assertTrue(Files.readString(Path.of("target/package/bin/bridge-stdio.bat"))
                .contains("weka-mcp-gui-0.8.0.jar"));
        if (!System.getProperty("os.name").toLowerCase().startsWith("windows")) {
            assertTrue(Files.isExecutable(Path.of("target/package/bin/bridge-stdio.sh")));
        }
        Properties description = new Properties();
        try (var reader = Files.newBufferedReader(Path.of("target/package/Description.props"))) {
            description.load(reader);
        }
        assertEquals("wekaMcp", description.getProperty("PackageName"));
        assertEquals("0.8.0", description.getProperty("Version"));
        assertEquals("weka (>=3.8.7)", description.getProperty("Depends"));
        Properties explorer = new Properties();
        try (var reader = Files.newBufferedReader(Path.of("target/package/Explorer.props"))) {
            explorer.load(reader);
        }
        assertEquals("weka.mcp.gui.McpPanel", explorer.getProperty("Tabs"));
        assertEquals("append", explorer.getProperty("TabsPolicy"));

        WekaPackageManager.installPackageFromArchive(archive.toString());
        assertTrue(Files.isRegularFile(WekaPackageManager.PACKAGES_DIR.toPath()
                .resolve("wekaMcp/tools/weka-mcp-java-1.6.0.jar")));
        WekaPackageManager.loadPackages(false, true, false);
        var installed = WekaPackageManager.getInstalledPackageInfo("wekaMcp");
        assertTrue(WekaPackageManager.loadCheck(installed, WekaPackageManager.getPackageHome()));
        assertTrue(WekaPackageManager.hasBeenLoaded(installed));
        ClassLoader packageLoader = weka.core.WekaPackageClassLoaderManager.getWekaPackageClassLoaderManager()
                .getLoaderForClass("weka.mcp.gui.McpPanel");
        assertNotNull(packageLoader, "Weka must load the panel from the installed package, not the test classpath");
        assertEquals(Class.forName("weka.mcp.gui.McpPanel", false, packageLoader),
                weka.core.WekaPackageClassLoaderManager.forName("weka.mcp.gui.McpPanel"));
        assertNotNull(Class.forName("weka.mcp.gui.McpPanel"));
        Explorer referenceExplorer = new Explorer();
        assertTrue(referenceExplorer.getPanels().stream()
                .anyMatch(panel -> panel.getClass().getName().equals("weka.mcp.gui.McpPanel")));
    }
}
