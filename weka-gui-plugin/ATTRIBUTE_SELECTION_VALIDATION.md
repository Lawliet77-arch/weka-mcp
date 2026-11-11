# Weka MCP GUI 0.6.0 原生属性选择验证报告

## 结论

在既有原生应用自动化桥接基础上，新增真实 Explorer **Select attributes 面板**的属性子集评估与排名，支持全量评估、交叉验证、结果读取和取消。

最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**14 个单元测试、3 个集成测试，合计 17 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.6.0.zip`
- SHA-256：`5558cb849a9404f78d6d965e7cf8af7cb3c899e8ab11f15626b4b9c624bdc662`
- 原生属性选择截图：`weka-gui-plugin/target/native-attribute-selection-result.png`
- 安装与 Agent 配置说明：安装包根目录 `README.md`
- 前期验证：`NATIVE_BRIDGE_VALIDATION.md`、`EXPLORER_DATA_VALIDATION.md`、`CLUSTER_VALIDATION.md`、`ASSOCIATION_VALIDATION.md`

## 新增工具

工具发现现返回 **43 个后台工具 + 18 个原生工具，共 61 个**。新增：

- `weka_explorer_select_attributes`：启动真实属性评估。
- `weka_explorer_get_attribute_selection_results`：等待并读取最新原生输出。
- `weka_explorer_stop_attribute_selection`：请求停止当前任务。

支持的组合：

- `CfsSubsetEval + BestFirst`
- `GainRatioAttributeEval + Ranker`
- `InfoGainAttributeEval + Ranker`

未指定 `folds` 时使用全量数据评估；指定 2～50 折时使用交叉验证，`seed` 仅用于交叉验证。`num_to_select` 仅用于 Ranker。评估器和搜索器分别接受 `evaluator_options`、`search_options`，错误搭配会明确拒绝。

`get_state` 新增 `attribute_selection_running`、`attribute_selection_result_count`、`attribute_evaluator` 和 `attribute_search`，属性选择任务也纳入 `task_running` 互斥。

## 实际验证

- CfsSubsetEval 在 Iris 全量数据上完成子集选择，结果包含 `Selected attributes` 和花瓣属性。
- GainRatioAttributeEval 在全量数据上按 Ranker 返回排名，限制返回数量为 3。
- InfoGainAttributeEval 使用 3 折交叉验证、种子 1，返回真实排名结果。
- 外部 Java MCP 客户端使用 GainRatio + Ranker、3 折交叉验证、种子 1，截图确认 Select attributes 页面显示 `Evaluation mode: 3-fold cross-validation` 和属性排名。
- 没有类别列时明确拒绝，未启动后台任务。
- 空闲时停止请求返回 `stop_requested=false`。
- 原有分类、聚类、关联、画像、撤销、后台分析、模型保存恢复和安装流程继续通过回归。

## 发现并修复的宿主界面问题

### 编辑器匹配提示造成阻塞

切换 GainRatio 与 Ranker 时，Weka 会弹出“是否改用 Ranker”或“是否改用 InfoGain”的确认框。两个方向都可能触发，单纯调整设置顺序不能解决自动化阻塞。

修复为在成对设置评估器和搜索器时：

1. 只暂停 `AttributeSelectionPanel` 自己注册的匹配提示监听器。
2. 设置匹配的评估器/搜索器组合。
3. 在 `finally` 中恢复原有监听器。

Weka `GenericObjectEditor` 不是 Swing 组件，因此通过其 `PropertyChangeSupport` 精确暂停和恢复监听，而不是屏蔽全局事件或隐藏评估异常。宿主拒绝配置时仍返回明确错误，不伪造结果。

### 类型与模式边界

- 评估器在 Weka 中实现 `OptionHandler`，按真实接口设置选项，不添加虚假的“无选项”分支。
- 结果返回 Weka 原生文本和模型可用性，尚未统一为结构化排名、 merit 或子集对象。
- 取消使用宿主中断机制；本版验证接口和空闲行为，未对所有长任务的中断时效作保证。
- 尚未接管任意对话框和可视化，也未覆盖 ReliefF、PCA 等更多评估方法。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未实测。
- 发布前仍需完成依赖许可证、版权通知及源码提供义务审查。
