# Weka MCP GUI 0.2.0 原生应用自动化桥接验证报告

## 结论

已实现并验证 **Agent → STDIO 连接器 → 本机安全桥接 → 当前 Weka Explorer** 的真实链路。桥接不再只是将分析结果写入一个新面板，而是可以读取和操作正在运行的 Explorer 数据、预处理、分类任务及结果列表。

GUI 0.2.0 最终执行 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**5 个单元测试、3 个集成测试，合计 8 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.2.0.zip`
- SHA-256：`2572f73ba47f6dccda3137d62b3cea602e9b1ccb24a8bf8c3cd6b7b88611cfe9`
- Agent 配置示例：`weka-gui-plugin/agent-config.example.json`
- 安装和配置说明：安装包根目录 `README.md`，源文件为 `src/main/package/README.md`
- 原生 Classify 验证截图：`weka-gui-plugin/target/native-explorer-result.png`

## 架构与边界

```text
Agent
  │ 标准 MCP JSON-RPC / STDIO
  ▼
bridge-stdio.sh / bridge-stdio.bat
  │ 回环 Socket、随机端口、随机令牌
  ▼
Explorer 内 Weka MCP 插件
  ├─ 7 个原生工具：当前 Explorer 的数据、预处理、分类、结果与取消
  └─ 原 43 个工具：按需启动独立 weka-mcp-java 1.6.0 子进程
```

连接器只加载 GUI JAR 和 Jackson。独立服务 fat JAR 仍只在子进程中运行，不作为 GUI 库加载。后台服务保留原 STDIO 协议；原生桥接通过单独的 STDIO 适配器供 Agent 连接，不占用 Weka 自己的标准输入输出。

**50 个可发现工具不全都操作原生窗口。** 7 个 `weka_explorer_*` 工具影响当前 Explorer；其余 43 个继续在独立后台引擎中执行。当前没有接管原生聚类、关联规则、属性选择、可视化或任意对话框。

## 原生操作验证

`NativeBridgeIT` 使用隔离 Weka 主目录安装真实 ZIP，创建可见 Explorer，并通过用户界面的“启动 Agent 桥接”按钮启用通道。随后由包内 `bin/bridge-stdio.sh` 启动独立 Java 连接器，发送真实 JSON-RPC 请求，而不是直接模拟工具返回。

已验证：

1. 初始化返回 `weka-mcp-explorer` 和协议版本 `2025-06-18`。
2. `tools/list` 返回 **50 个工具**，包含后台 `weka_train_classifier` 与原生 `weka_explorer_train_classifier`。
3. 读取当前 Iris 数据，确认 30 条实例及类别列 `class`。
4. 将类别列设置为第 5 列。
5. 从工作区已有文件加载 ARFF；单元测试还验证了从外部 ARFF 文件导入。
6. 对当前数据应用 `ReplaceMissingValues`，成功更新 Explorer 并保留撤销点。
7. 通过原生 Classify 页面执行 J48、3 折交叉验证、随机种子 1；读取到完整 Weka 输出。
8. 截图确认结果列表存在 `trees.J48`，正确分类 **28/30，93.3333%**，页面显示 Stratified cross-validation。
9. 启动 RandomForest、100000 棵树的长任务，确认停止请求被接受，随后分类线程不再运行。
10. 通过同一个外部 MCP 连接调用后台 `weka_load_dataset`，取得真实 `dataset_id`。
11. 错误访问令牌被拒绝；未知方法返回 JSON-RPC `-32601`。
12. 关闭标准输入后连接器退出码为 0；关闭桥接删除连接文件。

`ExplorerSessionTest` 另行验证真实 Explorer 的原生会话、数据导入、预处理、训练和结果读取。原工作台 `McpPanelWorkflowIT` 的安装、训练评估、预测、聚类、模型保存恢复及子进程退出回归继续通过。

## 安全与生命周期

- 桥接默认关闭，必须由用户在 Weka MCP 标签页显式启用。
- 仅绑定回环地址，使用随机端口和 256 位随机令牌；未提供正确的令牌不能建立 MCP 会话。
- POSIX 环境验证连接文件只允许所有者读写。
- 一次只服务一个 Agent 连接，后台工具调用串行化。
- 预处理成功后才更新 Explorer；若期间数据已被其他操作替换，拒绝覆盖新状态。
- 关闭桥接或 Explorer 会请求停止原生训练，关闭后台子进程并清理连接文件。
- 这是一套本机自动化接口，不是针对不可信调用方的安全沙箱，不应转发到远程机器。

## 本轮修复的问题

- 修复 Jackson 3 的节点处理，原生参数直接以 JSON 节点转发，避免数字、数组和对象经过不必要的 Java Map 转换。
- 修正 Weka 过滤器缺少 `setInputFormat` 导致的 `No input instance format defined`。
- 修正工作区内已有文件被错误转换为绝对路径的问题，统一使用数据根目录相对路径，再执行真实路径与越界检查。
- 修正 Maven 资源过滤位置，确保启动脚本中的 GUI 版本被实际替换。
- 使用 Weka 3.8.7 包中已确认的 Explorer 接口；分类器的少量受保护字段通过明确的版本适配器访问，失败时报告不支持，而不是假装成功。

## 验证范围与未覆盖项

- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`，该包运行时自报 `3.8.8-SNAPSHOT`。
- Windows `.bat` 已打包并检查版本替换，但没有在 Windows 上实际运行。Linux、其他 Weka 发行版和版本兼容矩阵仍需实测。
- 原生分类器当前允许 J48、RandomForest、NaiveBayes、Logistic、MultilayerPerceptron、ZeroR；过滤器为 ReplaceMissingValues、Normalize、Standardize、非监督/监督 Discretize。
- 原生结果返回 Weka 输出文本及模型可用性，尚未统一为结构化指标。
- 全量预处理发生在交叉验证之前，不能视为折内预处理；工具返回明确警告。严格评估应使用后台服务的受控预处理流程。
- 尚未实现原生聚类、关联规则、可视化、结构化数据画像、统一撤销接口及多客户端并发。它们不属于本轮已经验证的能力。
- 发布前仍需完成 fat JAR 中 Weka 和全部传递依赖的许可证、版权通知与源码提供义务审查。
