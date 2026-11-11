

# 企划书（修订版）：weka-mcp-java —— 纯 Java STDIO 嵌入式 Weka MCP 服务

## 1. 项目定位

构建一个**可执行 fat jar**，通过 **MCP STDIO 传输**向本地 Agent 暴露 Weka 的数据加载、预处理、分类器训练/评估、聚类和预测能力。Agent 以子进程方式启动该 jar，进程生命周期即会话生命周期。

**架构决策不变**：Weka 与 MCP Java SDK 运行在**同一 JVM 内**，零跨进程开销，不走 REST 中间层。**传输方式固定为 STDIO**，不包含任何 HTTP 服务端、认证、多租户或远程部署代码。这使 `DatasetRegistry` 和 `ModelRegistry` **天然是单会话的**，无需分布式状态管理，也无需考虑会话隔离。

## 2. 版本选型与依赖基线

这是方案成熟度的第一道门槛：版本必须锁在**稳定 GA 线**上，不能基于里程碑版本做架构假设。

| 组件             | 选型                                                                | 依据                                                                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP Java SDK** | `io.modelcontextprotocol.sdk:mcp` **1.0.x GA 线**（或 2.0.x GA 线） | 1.0.0 于 2026 年 2 月 GA，SDK 进入语义化版本稳定线，patch 版本仅做向后兼容修复。2.0.x 为当前主版本线，核心模块原生支持 STDIO，无需外部 Web 框架。**企划以 1.0.x GA 为最低基线**，2.0.x 可用但需验证 API 兼容性。 |
| **Weka**         | `nz.ac.waikato.cms.weka:weka-stable` **3.8.6+** 或 3.9.x            | 3.8.6+ 为长期稳定线；3.9.x 为开发线，若使用需锁定具体版本。Weka 的 Java API 可直接在服务端嵌入，支持训练、评估、预测全流程。                                                                                     |
| **JSON**         | Jackson 2 或 Jackson 3（SDK 1.0 默认 Jackson 3）                    | SDK 已解耦序列化绑定，可沿用项目现有 Jackson 版本。                                                                                                                                                              |
| **日志**         | SLF4J + Logback                                                     | **STDIO 模式下 stdout 必须只承载 JSON-RPC 协议帧**，所有日志必须写到 stderr 或文件。                                                                                                                             |
| **构建**         | Maven Shade 或 Gradle Shadow                                        | 产出单一 fat jar，`java -jar` 直接启动。                                                                                                                                                                         |
| **Java 版本**    | 17+                                                                 | Weka 3.8.6+ 和 SDK 1.0+ 均要求 Java 17 及以上。                                                                                                                                                                  |

**硬性约束**：不要在 stdio 模式下引入任何可能写入 `System.out` 的第三方库的默认配置（如 Spring Boot banner、某些日志框架的 ConsoleAppender 默认指向 stdout）。**启动阶段的第一项工程任务就是配置 Logback，将所有 ConsoleAppender 的目标设为 `System.err`**。

## 3. 启动时序（成熟方案的关键差异点）

一个“初期设想”可能只写“创建 server、注册 tool”，但成熟方案必须明确**初始化顺序**，否则会出现“部分分类器不可用”或“Weka 包未加载”的运行时故障。

### 3.1 强制启动顺序

```
1. 配置 Logback → stderr（在任何其他类加载之前）
2. 调用 WekaPackageManager.loadPackages(false, true, false)
3. 验证白名单中的每个类均可实例化
4. 构建 ModelRegistry 和 DatasetRegistry
5. 创建 StdioServerTransportProvider
6. 构建 McpSyncServer，设置 serverInfo 和 capabilities
7. 注册全部工具
8. 进入 stdio 事件循环（阻塞主线程）
```

**第 2 步不可省略**。Weka 的官方文档明确要求：在实例化任何 classifier、clusterer 或 filter 之前，必须调用 `WekaPackageManager.loadPackages(...)`，否则已安装的包（包括部分内置算法依赖的 MTJ/arpack 库）不会被加入 classpath。跳过这一步的典型症状是“`weka.classifiers.trees.RandomForest` 在 IDE 中可用，但 fat jar 中启动后抛出 `ClassNotFoundException`”。

**第 3 步是成熟方案的自检环节**：遍历白名单，对每个 `filter_class` / `classifier_class` / `clusterer_class` 尝试 `Class.forName` + 无参实例化 + `getOptions()`，记录失败项并在启动日志（stderr）中告警。这比运行到 Agent 调用时才报错要可靠得多。

### 3.2 主线程阻塞模型

`McpSyncServer` 基于阻塞式设计。fat jar 的 `main` 方法在完成注册后需要**显式阻塞主线程**（例如 `Thread.currentThread().join()` 或 `CountDownLatch.await()`），否则 JVM 可能在 stdio 事件循环就绪前退出。不要在 `main` 返回后依赖任何后台线程“碰巧”保持进程存活。

## 4. MCP 工具设计（成熟版）

### 4.1 工具清单

| 工具名                  | 功能                                    | 关键参数                                                        | 返回摘要                     |
| ----------------------- | --------------------------------------- | --------------------------------------------------------------- | ---------------------------- |
| `weka_load_dataset`     | 加载 ARFF/CSV，注册到 `DatasetRegistry` | `path`、`format`、`name`                                        | `dataset_id`、实例数、属性数 |
| `weka_list_filters`     | 列出白名单内的 Filter                   | 无                                                              | 类名 + 允许选项列表          |
| `weka_apply_filter`     | 应用 Filter，生成新数据集               | `dataset_id`、`filter_class`、`options`                         | 新 `dataset_id`、结构变化    |
| `weka_train_classifier` | 训练分类器，返回 `model_id`             | `dataset_id`、`classifier_class`、`options`、`class_index`      | `model_id`、训练摘要         |
| `weka_evaluate`         | 交叉验证或留出法评估                    | `dataset_id`、`classifier_class` 或 `model_id`、`folds`、`seed` | 准确率、Kappa、混淆矩阵      |
| `weka_predict`          | 用 `model_id` 预测                      | `model_id`、`dataset_id` 或 `instance_json`                     | 预测标签/分布                |
| `weka_cluster`          | 聚类                                    | `dataset_id`、`clusterer_class`、`options`、`num_clusters`      | `clusterer_id`、簇分配       |

### 4.2 工具规范化的硬约束

每个工具必须满足以下四条，否则不应进入白名单：

**窄而可解释**：工具名表达业务意图，不接受“万能参数”。`weka_train_classifier` 只做训练，不兼任评估或预测。拒绝 `execute_weka_code` 或类似反射入口。

**输入有界**：`dataset_id` 和 `model_id` 必须是注册表中存在的 UUID，拒绝任意字符串。`path` 必须解析为 `toRealPath()` 后仍位于配置的数据根目录内。`options` 数组的长度和单元素长度设上限。

**输出稳定**：成功时返回结构化 JSON 字符串；失败时返回 `CallToolResult` 并设置 `isError = true`，附可读错误消息，**不抛未捕获异常到 MCP 层**，以免客户端看到无结构的协议错误。

**选项可审计**：`filter_class` / `classifier_class` / `clusterer_class` 必须经过**显式白名单映射**，不允许 `Class.forName` 直接使用 Agent 传入的任意字符串。每个白名单类同时绑定一个**允许的 Weka 选项集合**，在调用 `setOptions(String[])` 之前逐项校验。

## 5. 有状态管理：单会话注册表

由于固定 STDIO，注册表的生命周期与进程完全一致，**无需 TTL、LRU 或跨会话隔离**，但仍需容量上限防止 Agent 意外加载过多数据集导致 OOM。

```java
public final class SessionRegistry {
    private static final int MAX_DATASETS = 20;
    private static final int MAX_MODELS = 50;

    private final Map<String, Instances> datasets = new LinkedHashMap<>(16, 0.75f, true);
    private final Map<String, Classifier> classifiers = new LinkedHashMap<>(16, 0.75f, true);
    private final Map<String, Clusterer> clusterers = new LinkedHashMap<>(16, 0.75f, true);

    public synchronized String registerDataset(Instances data) {
        if (datasets.size() >= MAX_DATASETS) {
            throw new IllegalStateException("数据集数量已达上限 " + MAX_DATASETS
                + "，请先释放不再使用的数据集");
        }
        String id = UUID.randomUUID().toString();
        datasets.put(id, data);
        return id;
    }
    // classifier / clusterer 同理
}
```

**成熟方案与初期设想的区别**在于：上限值以**配置项**形式暴露（环境变量或 `application.properties`），默认值保守（如数据集 20 个、模型 50 个），并在超限时返回**明确的、Agent 可理解的错误消息**，而不是静默 OOM。`LinkedHashMap(accessOrder=true)` 提供了隐式的 LRU 顺序，便于未来需要淘汰时直接扩展，但 STDIO 短会话场景下通常无需主动淘汰。

## 6. 安全加固（生产级）

### 6.1 白名单的维护机制

白名单不应硬编码在工具 handler 内部，而应外置为**可版本控制的配置文件**（如 `weka-whitelist.yaml`），启动时加载并完成第 3.1 节的自检。每个条目的结构：

```yaml
classifiers:
  - class: weka.classifiers.trees.J48
    allowed_options: ["-C", "-M", "-U"]
  - class: weka.classifiers.bayes.NaiveBayes
    allowed_options: []
filters:
  - class: weka.filters.unsupervised.attribute.Standardize
    allowed_options: []
  - class: weka.filters.unsupervised.attribute.Remove
    allowed_options: ["-R"]
```

`allowed_options` 是**选项标志（flag）**的集合，不含参数值。校验逻辑：遍历 Agent 传入的 `options` 数组，遇到 `-` 开头的 token 时检查是否在允许集合中；若不在，立即拒绝并返回“选项 `-X` 不在 `J48` 的允许列表中”。

### 6.2 文件路径约束

```java
Path allowedRoot = Path.of(config.getDataRoot()).toRealPath();
Path requested = Path.of(userPath).toRealPath(); // 解析符号链接
if (!requested.startsWith(allowedRoot)) {
    throw new SecurityException("路径超出允许的数据根目录");
}
```

使用 `toRealPath()` 而非 `normalize()`，因为前者会解析符号链接和 `..` 穿越，后者不会。

### 6.3 执行超时

每个可能长时间运行的工具（`train_classifier`、`evaluate`、`cluster`）必须包裹在**有界超时**中。由于 `McpSyncServer` 的 handler 是同步阻塞的，建议使用独立的 `ExecutorService` 提交训练任务，并通过 `Future.get(timeout, TimeUnit.SECONDS)` 施加边界。超时后 `cancel(true)` 并返回 `isError` 结果。

超时默认值建议：**训练/评估 120 秒，加载/过滤/预测 30 秒**。以配置项暴露，允许按运行环境调整。

### 6.4 资源上限

- **单数据集上限**：`maxInstances`（默认 100,000）和 `maxAttributes`（默认 10,000），在 `weka_load_dataset` 解析后立即检查。
- **选项数组上限**：`maxOptionsLength = 20`，单选项长度不超过 64 字符。
- **`instance_json` 上限**：预测工具接受的内联实例 JSON 字符串长度上限（默认 16 KB）。

## 7. 日志与可观测性

### 7.1 Logback 配置（必须最先加载）

```xml
<configuration>
    <appender name="STDERR" class="ch.qos.logback.core.ConsoleAppender">
        <target>System.err</target>
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>
    <root level="INFO">
        <appender-ref ref="STDERR"/>
    </root>
</configuration>
```

**验证方法**：在集成测试中加入一条断言——启动 fat jar 后，从 `stderr` 中应能看到启动日志，而 `stdout` 中**在发送第一个 MCP 请求之前应为空**。如果 stdout 出现任何非 JSON-RPC 内容，测试失败。

### 7.2 审计日志

每次工具调用在 `stderr` 输出一行结构化日志（JSON 或 key=value）：

```
tool=weka_train_classifier dataset_id=... classifier=weka.classifiers.trees.J48 options=[-C,0.25] elapsed_ms=1842 status=OK model_id=...
```

不记录完整数据集内容，不记录 `instance_json` 的原始值。审计日志是调试 Agent 行为和事后追溯的唯一依据。

## 8. 测试策略

成熟方案的测试矩阵至少覆盖以下四层：

| 层级             | 测试内容                                                                                           | 通过标准                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **协议纯净性**   | 启动 jar，发送一个最小 MCP 请求                                                                    | stdout 只出现合法 JSON-RPC 帧；stderr 有启动日志                |
| **工具契约**     | 对每个工具测试正常路径、缺失参数、非法 `dataset_id` / `model_id`、白名单外类名、非法选项、路径穿越 | 正常路径返回结构化 JSON；异常路径返回 `isError=true` 且消息可读 |
| **端到端工作流** | 加载 iris → 训练 J48 → 10 折交叉验证 → 预测                                                        | 全流程在一个 MCP 会话内完成，`model_id` 可复用                  |
| **资源边界**     | 加载超过 `maxInstances` 的数据集；训练超时；选项数组超长                                           | 被明确拒绝，错误消息指明触发的上限                              |

**Weka 自检测试**：在 CI 中对白名单配置执行一次“全部类可实例化”测试，防止依赖变更导致某个算法在打包后不可用。

## 9. 打包与部署

### 9.1 Fat Jar

Maven Shade Plugin 需注意 **Weka 的包资源合并**。Weka 的 `weka-stable` 中包含 `weka/experiment/DatabaseUtils.props`、`weka/core/version.txt` 等资源文件，Shade 默认可能覆盖或丢弃。需配置 `ServicesResourceTransformer` 和 `AppendingTransformer`，或使用 `maven-assembly-plugin` 的 `jar-with-dependencies` 并显式声明资源合并规则。**这是最容易在“初期设想”中被忽略、但在实际打包后导致启动失败的问题。**

### 9.2 启动脚本

提供 `start.sh` / `start.bat` 作为 MCP 客户端的 `command` 目标，封装 JVM 参数：

```bash
#!/usr/bin/env bash
exec java \
  -Xmx2g \
  -Dfile.encoding=UTF-8 \
  -Djava.awt.headless=true \
  -jar "$(dirname "$0")/weka-mcp-java-1.0.0.jar"
```

`-Djava.awt.headless=true` 是**必要的**：Weka 的部分类（如 Explorer 相关依赖）可能间接触发 AWT 初始化，无头模式下缺失该参数会导致不确定的启动失败。

### 9.3 MCP 客户端配置

```json
{
  "mcpServers": {
    "weka": {
      "command": "/path/to/start.sh"
    }
  }
}
```

不需要任何 `args` 或环境变量——数据根目录、超时、上限等通过 jar 同目录的 `config.properties` 或环境变量注入，保持客户端配置极简。

## 10. 交付物清单

一份成熟的 STDIO 方案应包含以下可交付项，而非仅仅是代码骨架：

| 交付物                    | 内容                                       |
| ------------------------- | ------------------------------------------ |
| **fat jar**               | `weka-mcp-java-1.0.0.jar`，含全部依赖      |
| **`weka-whitelist.yaml`** | 白名单及允许选项，与 jar 分离，可独立更新  |
| **`config.properties`**   | 数据根目录、超时、资源上限                 |
| **`logback.xml`**         | 嵌入 jar，stderr-only                      |
| **启动脚本**              | `start.sh` / `start.bat`                   |
| **集成测试套件**          | 覆盖第 8 节四层矩阵                        |
| **README**                | 客户端配置方式、白名单扩展方法、错误码含义 |

## 11. 与路径二的最终取舍（修订）

路径二（Node MCP + Java REST）的**唯一不可替代优势**是“MCP 层与 Weka 后端可独立部署”。用户已明确不需要远程能力，因此该优势**完全不适用**。路径一在 STDIO 单机场景下的优势是压倒性的：

- **零网络跳转**：训练和预测发生在同一进程，`model_id` 只是本地 Map 的 key，不存在 REST 序列化/反序列化开销和超时叠加。
- **零认证/授权复杂度**：STDIO 的信任边界是操作系统进程级，无需实现 OAuth、API Key 或 CIDR 策略。
- **部署只需一个 jar**：没有 Node 运行时依赖，没有端口分配，没有服务健康检查。

**结论**：在本用户的需求边界内，路径一是**唯一合理的工程选择**，不需要保留任何回退到路径二的迁移规划。

## 附：最小可执行代码骨架（关键片段）

```java
public class WekaMcpServer {
    public static void main(String[] args) throws Exception {
        // 1. 日志已由 logback.xml 导向 stderr（类加载前生效）
        // 2. Weka 包初始化
        WekaPackageManager.loadPackages(false, true, false);

        // 3. 加载白名单并自检
        Whitelist whitelist = WhitelistLoader.load("weka-whitelist.yaml");
        whitelist.selfCheck();

        // 4. 会话注册表
        SessionRegistry registry = new SessionRegistry(config);

        // 5. STDIO 传输
        var transport = new StdioServerTransportProvider(McpJsonDefaults.getMapper());

        // 6. MCP Server
        McpSyncServer server = McpServer.sync(transport)
            .serverInfo("weka-mcp-java", "1.0.0")
            .capabilities(ServerCapabilities.builder().tools(true).build())
            .build();

        // 7. 注册工具
        server.addTool(Tools.loadDataset(registry, config));
        server.addTool(Tools.trainClassifier(registry, whitelist, config));
        server.addTool(Tools.evaluate(registry, whitelist, config));
        server.addTool(Tools.predict(registry, config));

        // 8. 阻塞主线程
        Thread.currentThread().join();
    }
}
```

`SyncToolSpecification` 的 handler 签名与注册方式遵循 SDK 1.0 的稳定 API：`new SyncToolSpecification(tool, (exchange, arguments) -> CallToolResult...)`。在 handler 内部，所有异常被捕获并转换为 `isError=true` 的 `CallToolResult`，确保 Agent 始终收到结构化反馈而非协议层断裂。
