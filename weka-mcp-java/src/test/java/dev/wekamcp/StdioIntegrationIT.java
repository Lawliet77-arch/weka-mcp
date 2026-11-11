package dev.wekamcp;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class StdioIntegrationIT {
    private static final JsonMapper JSON = new JsonMapper();
    private static final Path PROJECT_ROOT = Path.of("").toAbsolutePath();

    @TempDir
    Path directory;

    @Test
    void fatJarSupportsACompleteMcpSession() throws Exception {
        Path iris = copyIrisWithCrLf(directory.resolve("data"));
        Path config = directory.resolve("config.properties");
        Files.writeString(config,
                "dataRoot=" + propertyPath(iris.getParent()) + "\n"
                        + "whitelistPath=" + propertyPath(PROJECT_ROOT.resolve("weka-whitelist.yaml")) + "\n"
                        + "searchSpacePath=" + propertyPath(PROJECT_ROOT.resolve("automl-search.yaml")) + "\n"
                        + "trainTimeoutSeconds=30\n");
        Path jar = jarPath();
        Process process = new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", javaExecutable()).toString(),
                "-Dfile.encoding=UTF-8",
                "-Djava.awt.headless=true",
                "-Dweka.mcp.config=" + config,
                "-jar", jar.toString())
                .start();
        verifyCompleteSession(process, iris);
    }

    @Test
    void startScriptSupportsAnIsolatedMcpDeployment() throws Exception {
        Path deployment = Files.createDirectories(directory.resolve("部署 环境"));
        Path iris = copyIrisWithCrLf(deployment.resolve("data"));
        Files.copy(jarPath(), deployment.resolve("weka-mcp-java-1.6.0.jar"), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(PROJECT_ROOT.resolve("config.properties"), deployment.resolve("config.properties"),
                StandardCopyOption.REPLACE_EXISTING);
        Files.copy(PROJECT_ROOT.resolve("weka-whitelist.yaml"), deployment.resolve("weka-whitelist.yaml"),
                StandardCopyOption.REPLACE_EXISTING);
        Files.copy(PROJECT_ROOT.resolve("automl-search.yaml"), deployment.resolve("automl-search.yaml"),
                StandardCopyOption.REPLACE_EXISTING);

        List<String> command;
        if (isWindows()) {
            Files.copy(PROJECT_ROOT.resolve("start.bat"), deployment.resolve("start.bat"),
                    StandardCopyOption.REPLACE_EXISTING);
            command = List.of("cmd.exe", "/d", "/s", "/c", "start.bat");
        } else {
            Files.copy(PROJECT_ROOT.resolve("start.sh"), deployment.resolve("start.sh"),
                    StandardCopyOption.REPLACE_EXISTING);
            command = List.of("/bin/bash", "start.sh");
        }
        Process process = new ProcessBuilder(command).directory(deployment.toFile()).start();
        verifyCompleteSession(process, iris);
    }

    private void verifyCompleteSession(Process process, Path iris) throws Exception {
        try (McpClient client = new McpClient(process)) {
            client.awaitStartup();
            assertEquals(0, process.getInputStream().available(), "stdout was not empty before the first MCP request");

            JsonNode initialized = client.request("initialize", Map.of(
                    "protocolVersion", "2025-06-18",
                    "capabilities", Map.of(),
                    "clientInfo", Map.of("name", "integration-test", "version", "1.0")));
            assertEquals("weka-mcp-java", initialized.get("serverInfo").get("name").asText());
            client.notify("notifications/initialized", Map.of());

            JsonNode tools = client.request("tools/list", Map.of());
            assertEquals(43, tools.get("tools").size());

            JsonNode loaded = client.toolSuccess("weka_load_dataset", Map.of(
                    "path", "iris.arff", "format", "arff", "class_index", 5));
            String datasetId = loaded.get("dataset_id").asText();
            Path renamed = iris.resolveSibling("iris-renamed.arff");
            Files.move(iris, renamed);

            JsonNode profile = client.toolSuccess("weka_profile_dataset", Map.of("dataset_id", datasetId));
            assertEquals(30, profile.get("instances").intValue());
            assertEquals(3, profile.get("class_distribution").size());

            JsonNode trained = client.toolSuccess("weka_train_classifier", Map.of(
                    "dataset_id", datasetId,
                    "classifier_class", "weka.classifiers.trees.J48"));
            String modelId = trained.get("model_id").asText();
            JsonNode explanation = client.toolSuccess("weka_explain_model", Map.of("model_id", modelId));
            assertTrue(explanation.get("supported").booleanValue());
            assertNotNull(explanation.get("dot"));
            JsonNode evaluation = client.toolSuccess("weka_evaluate", Map.of(
                    "dataset_id", datasetId, "model_id", modelId, "folds", 3));
            assertNotNull(evaluation.get("accuracy_percent"));

            String instance = JSON.writeValueAsString(Map.of(
                    "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
            JsonNode prediction = client.toolSuccess("weka_predict", Map.of(
                    "model_id", modelId, "instance_json", instance));
            assertEquals("Iris-setosa", prediction.get("predictions").get(0).get("predicted_value").asText());

            JsonNode automatic = client.toolSuccess("weka_auto_classify", Map.of(
                    "dataset_id", datasetId, "folds", 3));
            String automaticModelId = automatic.get("model_id").asText();
            assertTrue(automatic.get("auto_preprocess").booleanValue());
            assertEquals(3, automatic.get("evaluation").get("per_class").size());
            JsonNode independent = client.toolSuccess("weka_evaluate_model", Map.of(
                    "model_id", automaticModelId, "test_dataset_id", datasetId));
            assertTrue(independent.get("independent_test").booleanValue());
            JsonNode resources = client.toolSuccess("weka_list_resources", Map.of());
            assertEquals(2, resources.get("model_count").intValue());

            JsonNode recommendations = client.toolSuccess("weka_recommend", Map.of("dataset_id", datasetId));
            assertTrue(recommendations.get("requires_validation").booleanValue());
            assertEquals(4, client.toolSuccess("weka_histogram", Map.of(
                    "dataset_id", datasetId, "attribute_index", 1, "bins", 4)).get("bins").size());
            assertEquals(3, client.toolSuccess("weka_scatter_matrix", Map.of(
                    "dataset_id", datasetId, "attributes", List.of(1, 2, 3), "max_points", 5))
                    .get("plot_count").intValue());
            JsonNode split = client.toolSuccess("weka_split_dataset", Map.of(
                    "dataset_id", datasetId, "train_percent", 80, "seed", 42));
            assertEquals(24, split.get("training_instances").intValue());
            assertEquals(6, split.get("test_instances").intValue());
            assertEquals(30, client.toolSuccess("weka_export_dataset", Map.of(
                    "dataset_id", datasetId, "path", "exports/iris.csv", "format", "csv"))
                    .get("instances").intValue());
            assertEquals(30, client.toolSuccess("weka_load_dataset", Map.of(
                    "path", "exports/iris.csv", "format", "csv")).get("instances").intValue());
            assertEquals(2, client.toolSuccess("weka_transform_preview", Map.of(
                    "dataset_id", datasetId, "filter_class", "weka.filters.unsupervised.attribute.Normalize",
                    "max_rows", 2)).get("rows").size());

            JsonNode comparison = client.toolSuccess("weka_compare_models", Map.of(
                    "dataset_id", datasetId, "model_ids", List.of(modelId, automaticModelId),
                    "folds", 2, "runs", 1, "seed", 42));
            assertTrue(comparison.get("shared_folds").booleanValue());
            assertTrue(comparison.get("comparisons").get(0).get("p_value").doubleValue() >= 0.0);
            assertTrue(comparison.get("comparisons").get(0).get("p_value").doubleValue() <= 1.0);

            Files.writeString(iris.resolveSibling("binary.arff"), """
                    @relation binary
                    @attribute x numeric
                    @attribute class {a,b}
                    @data
                    0,a
                    0.1,a
                    0.2,a
                    0.3,a
                    1,b
                    1.1,b
                    """);
            String binaryId = client.toolSuccess("weka_load_dataset", Map.of("path", "binary.arff"))
                    .get("dataset_id").asText();
            String binaryModel = client.toolSuccess("weka_train_classifier", Map.of(
                    "dataset_id", binaryId, "classifier_class", "weka.classifiers.bayes.NaiveBayes"))
                    .get("model_id").asText();
            JsonNode thresholds = client.toolSuccess("weka_threshold_analysis", Map.of(
                    "model_id", binaryModel, "dataset_id", binaryId, "positive_class", "b"));
            assertEquals(21, thresholds.get("thresholds").size());
            assertEquals(6, thresholds.get("scored_instances").intValue());
            assertEquals(0, thresholds.get("minimum_cost").doubleValue(), 0.0);

            JsonNode ignoredOptions = client.callTool("weka_evaluate", Map.of(
                    "dataset_id", datasetId,
                    "model_id", modelId,
                    "options", List.of("-U")));
            assertTrue(ignoredOptions.get("isError").booleanValue());

            JsonNode nonFinite = client.callTool("weka_predict", Map.of(
                    "model_id", modelId,
                    "instance_json", instance.replace("5.1", "\"NaN\"")));
            assertTrue(nonFinite.get("isError").booleanValue());
            assertTrue(nonFinite.toString().contains("finite"));

            JsonNode traversal = client.callTool("weka_load_dataset", Map.of("path", "../iris.arff"));
            assertTrue(traversal.get("isError").booleanValue());
            client.finish();
            assertTrue(client.stderr().contains("started with 43 tools"));
        }
    }

    @Test
    void experimentModelRestoresAcrossJvmRestart() throws Exception {
        Path iris = copyIrisWithCrLf(directory.resolve("data"));
        Path config = directory.resolve("config.properties");
        Files.writeString(config,
                "dataRoot=" + propertyPath(iris.getParent()) + "\n"
                        + "experimentRoot=" + propertyPath(directory.resolve("experiments")) + "\n"
                        + "whitelistPath=" + propertyPath(PROJECT_ROOT.resolve("weka-whitelist.yaml")) + "\n"
                        + "searchSpacePath=" + propertyPath(PROJECT_ROOT.resolve("automl-search.yaml")) + "\n"
                        + "trainTimeoutSeconds=30\n");
        String instance = JSON.writeValueAsString(Map.of(
                "sepallength", 5.1, "sepalwidth", 3.5, "petallength", 1.4, "petalwidth", 0.2));
        String experimentId;
        String ordinaryExperiment;
        String streamExperiment;
        String clustererExperiment;
        JsonNode expectedPrediction;
        JsonNode ordinaryPrediction;
        JsonNode streamPrediction;
        JsonNode expectedCluster;
        Process first = startJavaProcess(config);
        try (McpClient client = new McpClient(first)) {
            client.awaitStartup();
            client.request("initialize", Map.of(
                    "protocolVersion", "2025-06-18", "capabilities", Map.of(),
                    "clientInfo", Map.of("name", "restart-test-1", "version", "1.0")));
            client.notify("notifications/initialized", Map.of());
            String datasetId = client.toolSuccess("weka_load_dataset", Map.of(
                    "path", "iris.arff", "class_index", 5)).get("dataset_id").asText();
            JsonNode automatic = client.toolSuccess("weka_auto_classify", Map.of(
                    "dataset_id", datasetId, "search", "halving", "protocol", "nested_cv",
                    "candidates", List.of(
                            Map.of("classifier_class", "weka.classifiers.trees.RandomTree", "options", List.of("-K", "2")),
                            Map.of("classifier_class", "weka.classifiers.bayes.NaiveBayes")),
                    "outer_folds", 2, "inner_folds", 2, "seed", 42, "budgetSeconds", 120));
            assertEquals(3, automatic.get("halving_stages").size());
            assertTrue(automatic.get("nested_cv_complete").booleanValue());
            experimentId = automatic.get("experiment_id").asText();
            expectedPrediction = client.toolSuccess("weka_predict", Map.of(
                    "model_id", automatic.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0);

            JsonNode ordinary = client.toolSuccess("weka_train_classifier", Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.trees.J48"));
            ordinaryPrediction = client.toolSuccess("weka_predict", Map.of(
                    "model_id", ordinary.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0);
            JsonNode saved = client.toolSuccess("weka_save_model", Map.of("model_id", ordinary.get("model_id").asText()));
            assertEquals(ordinary.get("model_id"), saved.get("model_id"));
            ordinaryExperiment = saved.get("experiment_id").asText();
            assertTrue(client.toolSuccess("weka_export_model", Map.of(
                    "model_id", ordinary.get("model_id").asText(), "path", "exports/j48.model"))
                    .get("gui_compatible").booleanValue());
            JsonNode imported = client.toolSuccess("weka_import_model", Map.of(
                    "path", "exports/j48.model", "dataset_id", datasetId, "trusted", true));
            assertFalse(imported.get("retrainable").booleanValue());
            assertEquals(ordinaryPrediction, client.toolSuccess("weka_predict", Map.of(
                    "model_id", imported.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0));
            String logistic = client.toolSuccess("weka_train_classifier", Map.of(
                    "dataset_id", datasetId, "classifier_class", "weka.classifiers.functions.Logistic"))
                    .get("model_id").asText();
            assertEquals("pmml", client.toolSuccess("weka_export_model", Map.of(
                    "model_id", logistic, "path", "exports/logistic.pmml", "format", "pmml"))
                    .get("format").asText());

            JsonNode stream = client.toolSuccess("weka_train_stream", Map.of(
                    "path", "iris.arff", "classifier_class", "weka.classifiers.bayes.NaiveBayesUpdateable",
                    "class_index", 5));
            assertEquals(30, stream.get("read_instances").intValue());
            assertEquals(24, stream.get("training_instances").intValue());
            assertEquals(6, stream.get("test_instances").intValue());
            streamExperiment = stream.get("experiment_id").asText();
            Files.writeString(iris.resolveSibling("update.arff"), """
                    @relation iris
                    @attribute sepallength numeric
                    @attribute sepalwidth numeric
                    @attribute petallength numeric
                    @attribute petalwidth numeric
                    @attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}
                    @data
                    5.1,3.5,1.4,0.2,Iris-setosa
                    6.2,3.4,5.4,2.3,Iris-virginica
                    """);
            JsonNode updated = client.toolSuccess("weka_update_model", Map.of(
                    "model_id", stream.get("model_id").asText(), "path", "update.arff"));
            assertEquals(2, updated.get("updated_instances").intValue());
            streamExperiment = updated.get("experiment_id").asText();
            streamPrediction = client.toolSuccess("weka_predict", Map.of(
                    "model_id", stream.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0);

            JsonNode clusterer = client.toolSuccess("weka_cluster", Map.of(
                    "dataset_id", datasetId, "clusterer_class", "weka.clusterers.SimpleKMeans",
                    "num_clusters", 3, "persist", true));
            clustererExperiment = clusterer.get("experiment_id").asText();
            expectedCluster = client.toolSuccess("weka_cluster_predict", Map.of(
                    "clusterer_id", clusterer.get("clusterer_id").asText(), "instance_json", instance))
                    .get("predictions").get(0);
            client.finish();
        }

        Process second = startJavaProcess(config);
        try (McpClient client = new McpClient(second)) {
            client.awaitStartup();
            client.request("initialize", Map.of(
                    "protocolVersion", "2025-06-18", "capabilities", Map.of(),
                    "clientInfo", Map.of("name", "restart-test-2", "version", "1.0")));
            client.notify("notifications/initialized", Map.of());
            JsonNode restored = client.toolSuccess("weka_load_experiment_model", Map.of(
                    "experiment_id", experimentId));
            JsonNode actualPrediction = client.toolSuccess("weka_predict", Map.of(
                    "model_id", restored.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0);
            assertEquals(expectedPrediction, actualPrediction);
            assertEquals(2, client.toolSuccess("weka_get_experiment", Map.of("experiment_id", experimentId))
                    .get("search").get("outer_folds").size());
            assertEquals("full_data", client.toolSuccess("weka_get_experiment", Map.of("experiment_id", experimentId))
                    .get("search").get("selection_source").asText());

            JsonNode ordinary = client.toolSuccess("weka_load_experiment_model", Map.of(
                    "experiment_id", ordinaryExperiment));
            assertEquals(ordinaryPrediction, client.toolSuccess("weka_predict", Map.of(
                    "model_id", ordinary.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0));

            JsonNode stream = client.toolSuccess("weka_load_experiment_model", Map.of(
                    "experiment_id", streamExperiment));
            assertEquals(streamPrediction, client.toolSuccess("weka_predict", Map.of(
                    "model_id", stream.get("model_id").asText(), "instance_json", instance))
                    .get("predictions").get(0));
            JsonNode streamManifest = client.toolSuccess("weka_get_experiment", Map.of(
                    "experiment_id", streamExperiment));
            assertEquals("update.arff", streamManifest.get("sourcePath").asText());
            assertEquals("incremental_update", streamManifest.get("protocol").asText());
            assertEquals("UPDATE_SOURCE", streamManifest.get("datasetRole").asText());
            assertFalse(Files.exists(directory.resolve("experiments").resolve(streamExperiment).resolve("dataset.arff")));

            JsonNode clusterer = client.toolSuccess("weka_load_clusterer", Map.of(
                    "experiment_id", clustererExperiment));
            assertEquals(expectedCluster, client.toolSuccess("weka_cluster_predict", Map.of(
                    "clusterer_id", clusterer.get("clusterer_id").asText(), "instance_json", instance))
                    .get("predictions").get(0));
            JsonNode savedClusterer = client.toolSuccess("weka_save_clusterer", Map.of(
                    "clusterer_id", clusterer.get("clusterer_id").asText()));
            assertFalse(savedClusterer.get("experiment_id").asText().isBlank());
            client.finish();
        }
    }

    private static Process startJavaProcess(Path config) throws IOException {
        return new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", javaExecutable()).toString(),
                "-Dfile.encoding=UTF-8",
                "-Djava.awt.headless=true",
                "-Dweka.mcp.config=" + config,
                "-jar", jarPath().toString())
                .start();
    }

    private static Path copyIrisWithCrLf(Path dataDirectory) throws IOException {
        Path data = Files.createDirectories(dataDirectory);
        Path iris = data.resolve("iris.arff");
        try (InputStream input = StdioIntegrationIT.class.getResourceAsStream("/iris.arff")) {
            assertNotNull(input);
            Files.copy(input, iris, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.writeString(iris, Files.readString(iris).replace("\n", "\r\n"));
        return iris;
    }

    private static Path jarPath() {
        Path jar = Path.of(System.getProperty("weka.mcp.jar")).toAbsolutePath();
        assertTrue(Files.isRegularFile(jar), "Fat jar was not built: " + jar);
        return jar;
    }

    private static String propertyPath(Path path) {
        return path.toAbsolutePath().toString().replace('\\', '/');
    }

    private static String javaExecutable() {
        return isWindows() ? "java.exe" : "java";
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase(Locale.ROOT).startsWith("windows");
    }

    private static final class McpClient implements AutoCloseable {
        private final Process process;
        private final BufferedWriter input;
        private final BufferedReader output;
        private final BufferedReader error;
        private final ExecutorService readerExecutor;
        private final StringBuilder stderr = new StringBuilder();
        private int sequence;
        private boolean finished;

        private McpClient(Process process) throws IOException {
            this.process = process;
            this.input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            this.output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            this.error = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
            this.readerExecutor = Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "mcp-test-reader");
                thread.setDaemon(true);
                return thread;
            });
            Thread stderrThread = new Thread(this::drainStderr, "mcp-test-stderr");
            stderrThread.setDaemon(true);
            stderrThread.start();
        }

        private void awaitStartup() throws InterruptedException {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
            while (System.nanoTime() < deadline) {
                if (stderr().contains("started with 43 tools")) {
                    return;
                }
                if (!process.isAlive()) {
                    fail("MCP process exited during startup with code " + process.exitValue() + "; " + stderr());
                }
                Thread.sleep(25);
            }
            fail("Timed out waiting for MCP startup; " + stderr());
        }

        private JsonNode request(String method, Map<String, Object> parameters) throws Exception {
            int id = ++sequence;
            ObjectNode request = JSON.createObjectNode();
            request.put("jsonrpc", "2.0");
            request.put("id", id);
            request.put("method", method);
            request.set("params", JSON.valueToTree(parameters));
            send(request);
            JsonNode response = readResponse();
            assertEquals(id, response.get("id").intValue());
            if (response.has("error")) {
                fail("MCP request failed: " + response);
            }
            return response.get("result");
        }

        private void notify(String method, Map<String, Object> parameters) throws IOException {
            ObjectNode notification = JSON.createObjectNode();
            notification.put("jsonrpc", "2.0");
            notification.put("method", method);
            notification.set("params", JSON.valueToTree(parameters));
            send(notification);
        }

        private JsonNode callTool(String name, Map<String, Object> arguments) throws Exception {
            return request("tools/call", Map.of("name", name, "arguments", arguments));
        }

        private JsonNode toolSuccess(String name, Map<String, Object> arguments) throws Exception {
            JsonNode result = callTool(name, arguments);
            assertFalse(result.get("isError").booleanValue(), () -> result.toString());
            String text = result.get("content").get(0).get("text").asText();
            return JSON.readTree(text);
        }

        private void send(JsonNode message) throws IOException {
            input.write(JSON.writeValueAsString(message));
            input.newLine();
            input.flush();
        }

        private JsonNode readResponse() throws Exception {
            Future<String> future = readerExecutor.submit(output::readLine);
            String line = future.get(30, TimeUnit.SECONDS);
            assertNotNull(line, () -> "MCP process closed stdout; stderr=" + stderr());
            JsonNode response = JSON.readTree(line);
            assertEquals("2.0", response.get("jsonrpc").asText());
            return response;
        }

        private void drainStderr() {
            try {
                String line;
                while ((line = error.readLine()) != null) {
                    synchronized (stderr) {
                        stderr.append(line).append('\n');
                    }
                }
            } catch (IOException ignored) {
            }
        }

        private String stderr() {
            synchronized (stderr) {
                return stderr.toString();
            }
        }

        private void finish() throws Exception {
            input.close();
            assertTrue(process.waitFor(15, TimeUnit.SECONDS), () -> "MCP process did not exit on stdin EOF; " + stderr());
            assertEquals(0, process.exitValue());
            finished = true;
        }

        @Override
        public void close() throws Exception {
            if (!finished) {
                process.descendants().forEach(ProcessHandle::destroyForcibly);
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
            }
            readerExecutor.shutdownNow();
        }
    }
}
