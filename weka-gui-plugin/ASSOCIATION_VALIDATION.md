# Weka MCP GUI 0.5.0 原生关联分析验证报告

## 结论

在既有原生应用自动化桥接基础上，新增真实 Explorer **Associate 面板的 Apriori 与 FPGrowth 关联分析**，支持结果读取、取消和统一任务状态检查。

最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**11 个单元测试、3 个集成测试，合计 14 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.5.0.zip`
- SHA-256：`97ddc524e67c49ad4febfffe4781589e9817dac3dea9cb50f556797390b139b5`
- 原生关联分析截图：`weka-gui-plugin/target/native-association-result.png`
- 安装与 Agent 配置说明：安装包根目录 `README.md`
- 前期验证：`NATIVE_BRIDGE_VALIDATION.md`、`EXPLORER_DATA_VALIDATION.md`、`CLUSTER_VALIDATION.md`

## 新增工具

工具发现现返回 **43 个后台工具 + 15 个原生工具，共 58 个**。新增：

- `weka_explorer_run_association`：启动真实 Associate 面板任务，默认 FPGrowth，也支持 Apriori。
- `weka_explorer_get_association_results`：等待并读取最新原生输出。
- `weka_explorer_stop_association`：请求停止当前关联任务。

`num_rules` 可选范围为 1～100，未指定时使用 Weka 默认值；其他参数通过 `OptionHandler` 传递。`get_state` 增加 `associator_running`、`association_result_count`、`association_data_scope` 和 `association_attributes`，并将关联任务纳入 `task_running` 互斥。

## 算法验证

### Apriori 与 FPGrowth

使用真实 Explorer 和 14 条 weather 名义数据，分别运行 Apriori 和 FPGrowth，均获得完成的原生结果，输出包含 `play` 属性。完成时 `associator_running=false`，当前标签为 `Associate`，结果数正确，空闲时停止请求返回 `stop_requested=false`。

外部 Java MCP 客户端还执行了以下完整流程：

1. 对 Iris 数值属性应用 Discretize。
2. 通过真实 STDIO 连接器启动 FPGrowth，规则数上限为 10。
3. 读取到实际关联规则，截图显示 **FPGrowth found 12 rules (displaying top 10)**。
4. 验证原始 Preprocess 仍为 5 个属性，而 Associate 使用了更多编码属性，没有替换原数据。

未直接传入数值属性时，工具会提示先执行 Discretize，而不是把类型不兼容误报为算法内部错误。

## 发现并修复的问题

### 多值名义属性不兼容

该 Weka 版本的 FPGrowth 仅支持一元或二元名义属性，weather 和离散化 Iris 的多值名义属性导致面板不启动，最初没有结果或错误文本。

修复为在 **Associate 面板内**生成二元名义数据：

- 每个多值名义值生成一个 `属性=值` 的二元属性。
- 保留一元/二元属性及缺失值。
- 展开最多 2000 个属性，防止无界膨胀。
- 只改变 Associate 的输入，不改变 Preprocess 原数据。
- 通过 `association_data_scope`、`association_attributes` 和 `binary_nominal_expansion` 明确返回转换范围。

如果宿主仍拒绝配置，现在返回明确错误，不再静默产生空结果。

### 会话元数据重置

外部 MCP 测试发现桥接每次调用都创建新会话，导致编码范围被重置为 `current_dataset`。现已改为在桥接生命周期内复用同一个 `ExplorerSession`，使训练范围、结果和写入互斥状态连续。关闭桥接时也通过同一个会话清理分类、聚类和关联任务。

## 回归与边界

原有分类、聚类、画像、撤销、后台分析、安装包加载、模型保存恢复、启动脚本和进程清理均通过回归。

- 关联结果目前返回 Weka 原生文本及模型可用性，未统一为结构化规则、支持度、置信度或提升度对象。
- 停止接口使用 Weka 的中断机制；本版验证接口和空闲行为，未对所有长关联任务的中断时效作保证。
- 尚未接管属性选择、任意对话框和可视化，也没有多客户端并发支持。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未实测。
- 发布前仍需完成依赖许可证、版权通知及源码提供义务审查。
