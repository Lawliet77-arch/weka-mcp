# Weka MCP GUI 0.3.0 原生数据画像与撤销验证报告

## 结论

在 0.2.0 原生应用自动化桥接基础上，新增 **原生数据画像**和 **Preprocess 数据撤销**。Agent 可读取当前 Explorer 数据的结构化统计，并在预处理后安全恢复数据。

最终执行 `mvn -B -ntp clean verify`：**BUILD SUCCESS**。共 **6 个单元测试、3 个集成测试，合计 9 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.3.0.zip`
- SHA-256：`996417e539ef057f02a1eeff6f0c3fe10427a9b7e51d1a228f5711ca658ff0b2`
- 使用说明：安装包根目录 `README.md`
- Agent 配置示例：`weka-gui-plugin/agent-config.example.json`
- 0.2.0 桥接基础验证：`weka-gui-plugin/NATIVE_BRIDGE_VALIDATION.md`
- 原生结果截图：`weka-gui-plugin/target/native-explorer-result.png`

## 新增工具

工具发现现返回 **43 个后台工具 + 9 个原生工具，共 52 个**。新增两项为：

- `weka_explorer_describe_dataset`：返回关系名、实例数、属性数、总权重、类别列、缺失总数，以及每个属性的索引、名称、类型、缺失数、不同值数、唯一值数。
- 数值属性额外返回最小值、最大值、均值和标准差；名义属性返回有上限的值列表及截断标记。
- 默认最多返回 100 个属性、每个名义属性 20 个值，可分别调整为 1000、100。属性统计针对调用开始时的快照。
- `weka_explorer_undo`：调用真实 Explorer Preprocess 撤销，等待原生异步 I/O 完成，再返回当前状态及是否发生变化。

撤销只作用于数据，不删除模型或历史结果。分类任务运行中或预处理 I/O 忙时会拒绝撤销；存在历史结果时返回 `existing_results_may_be_stale`，避免把旧评估误认为新数据的评估。

## 实际验证

### 数据画像

使用真实 Explorer 中的 Iris 数据（30 条、5 个属性）：

- 返回 5 个属性和缺失总数 0。
- 数值属性存在均值等统计字段。
- 类别属性有 3 个名义值，将上限设为 2 后只返回 2 个，并明确 `values_truncated=true`。
- 外部 STDIO 客户端通过真实本地桥接取得同样结果，不是直接调用伪造的统计对象。

### 撤销与恢复

1. 记录第一个数值属性的原始最小值。
2. 使用 Normalize，并显式设置输出范围为 0～1。
3. 验证归一化后最小值为 0、最大值为 1。
4. 调用原生撤销，确认 `changed=true`、类别索引保持为 5。
5. 再次画像，最小值恢复到原始值。

另外验证空 Explorer、没有撤销历史时，工具返回 `changed=false`、`dataset_loaded=false`，不会误报已恢复。

### 全链路回归

`NativeBridgeIT` 使用真实安装包、可见 Explorer、随包 `bridge-stdio.sh` 和外部 Java MCP 客户端，验证：

- 工具发现数量为 52。
- 数据加载、类别列设置、预处理、画像和撤销可连续执行。
- J48 三折交叉验证仍返回 28/30、93.3333%，原生 Classify 截图已人工查看。
- RandomForest 长任务可取消。
- 后台 `weka_load_dataset` 仍返回真实数据集 ID。
- 错误令牌拒绝、连接器退出和桥接文件清理正常。

原有 `McpPanelWorkflowIT` 与 `PackageInstallationIT` 继续通过，覆盖工作台操作、模型保存恢复、ZIP 安装、类加载、脚本版本替换与执行权限。

## 实现边界

- 原生 I/O 撤销由 Weka 异步执行；实现等待对应线程结束并刷新 EDT 状态后才返回，而不是调用 `undo()` 后立即宣称完成。
- 画像限制返回属性数和名义值数，避免默认无界输出；完整缺失计数仍针对整个数据快照计算。
- 撤销复用 Weka 自身撤销点和数据加载机制，不承诺撤销模型训练、分类结果或用户在其他对话框中的操作。
- 尚未实现原生聚类、关联规则、属性选择、可视化、结构化分类指标、多客户端并发及完整的数据版本血缘。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未在 Windows 上实测。
- 发布前仍需完成所有传递依赖的许可证、版权通知及源码提供义务审查。
