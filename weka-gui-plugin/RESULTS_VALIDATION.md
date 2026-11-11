# Weka MCP GUI 0.8.0 结构化结果与数据版本验证报告

## 结论

在既有原生应用自动化桥接基础上，新增 **结构化分类/聚类指标、操作 ID、训练/当前数据快照、取消状态及旧数据识别**。Agent 可以保留原始 Weka 输出，同时判断结果是否完成、取消或已不适用于当前数据。

最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**18 个单元测试、3 个集成测试，合计 20 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.8.0.zip`
- SHA-256：`f0a4891628129fb05f424a075d61030b808541594522d354e94dc2a3cbd3e836`
- 安装与 Agent 配置说明：安装包根目录 `README.md`
- 前期验证：`NATIVE_BRIDGE_VALIDATION.md`、`EXPLORER_DATA_VALIDATION.md`、`CLUSTER_VALIDATION.md`、`ASSOCIATION_VALIDATION.md`、`ATTRIBUTE_SELECTION_VALIDATION.md`、`VISUALIZATION_VALIDATION.md`

## 新增结果字段

工具总数保持 **43 个后台工具 + 21 个原生工具，共 64 个**，本版增强返回结构，不新增工具数量。

- `operation_id`：桥接创建的本地原生操作 ID，区别于后台服务的实验 ID。
- `training_dataset`：训练开始时的数据快照元数据。
- `current_dataset`：读取结果时的数据快照元数据。
- `dataset_changed_since_training`：当前数据对象是否已不同于训练时。
- `result_status`：`RUNNING`、`COMPLETE`、`STALE_DATASET`、`CANCELLED`、`INCOMPLETE`、`UNTRACKED_HISTORY`。
- `evaluation_metrics`：分类和聚类的关键结构化指标。

`get_state` 同时返回当前 `dataset_version`。快照只包含关系名、实例数、属性数和类别列，不包含原始数据行。

**`dataset_version` 是当前会话中的对象身份标记，不是内容哈希。** 它用于识别 Explorer 数据对象替换，不保证检测所有原地修改，也不构成跨进程持久化血缘。`COMPLETE` 只表示计算结束，不表示结果仍适用于当前数据；未被桥接跟踪的历史结果标为 `UNTRACKED_HISTORY`，不伪造版本信息。

## 指标与状态验证

### 分类指标

真实 Explorer 的 J48 结果现在包含：

- `correctly_classified`、`incorrectly_classified` 及各自 `_percent`。
- `kappa`。
- `mean_absolute_error`、`root_mean_squared_error`。
- `relative_absolute_error_percent`、`root_relative_squared_error_percent`。
- `total_instances`。

验证期间发现同一行同时包含样本数和百分比，已修正为分别读取两列。`ExplorerResultMetadataTest` 明确验证样本数 28 与百分比 93.3333 不会混淆，Kappa、MAE、RMSE、相对误差和总实例数也正确解析。

### 聚类指标

SimpleKMeans 结果包含 `within_cluster_squared_error`，数值来自真实 Weka 输出。没有为不支持的操作虚构统一指标。

### 完成、取消与过期

- 正常完成的分类、聚类、关联和属性选择返回 `COMPLETE`，且 `dataset_changed_since_training=false`。
- 取消 RandomForest 分类和 SimpleKMeans 聚类长任务后，返回 `CANCELLED`，不再显示为正常完成。
- 先完成 J48，再修改数据集，保留原 `operation_id`，但状态变为 `STALE_DATASET`，`dataset_changed_since_training=true`，训练与当前快照版本不同。
- 任务运行时优先返回当前任务的部分输出，不把上一次历史结果冒充当前结果。
- 历史计数记录任务启动前的位置，避免快速任务把已有历史误判为本次完成。

## 全链路回归

- 外部 Java MCP 客户端通过真实 STDIO 验证指标、操作 ID、完成状态、取消状态及原始结果文本。
- 分类、聚类、关联、属性选择、画像、撤销、原生散点图、后台分析、模型保存恢复、ZIP 安装和进程清理全部继续通过。
- 启动脚本中的 GUI 版本已替换为 0.8.0，并验证可执行权限及实际调用。

## 边界

- 关联规则、属性排名和模型内部结构尚未全部结构化；这些结果继续返回原生文本。
- 专用分类误差、聚类分配及任意第三方对话框的可视化尚未覆盖。
- 当前只支持一个本地桥接客户端，不保证多客户端并发操作。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未实测。
- 发布前仍需完成依赖许可证、版权通知及源码提供义务审查。
