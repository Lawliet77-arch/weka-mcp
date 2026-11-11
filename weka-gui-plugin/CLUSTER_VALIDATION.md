# Weka MCP GUI 0.4.0 原生聚类与任务互斥验证报告

## 结论

在既有原生应用自动化桥接基础上，新增真实 Explorer **Cluster 面板聚类、结果读取和取消**，并统一分类、聚类和预处理的忙碌状态检查。

最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**9 个单元测试、3 个集成测试，合计 12 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.4.0.zip`
- SHA-256：`c7f16c538aa2a897132200b109274010624b68e10f11b5cb810459d1a9ff66f0`
- 原生聚类截图：`weka-gui-plugin/target/native-cluster-result.png`
- 安装与 Agent 配置说明：安装包根目录 `README.md`
- 前期桥接验证：`NATIVE_BRIDGE_VALIDATION.md`、`EXPLORER_DATA_VALIDATION.md`

## 新增原生能力

工具发现现返回 **43 个后台工具 + 12 个原生工具，共 55 个**。新增：

- `weka_explorer_train_clusterer`：在当前 Explorer 的 Cluster 页面启动聚类。
- `weka_explorer_get_cluster_results`：等待并读取原生聚类结果列表中的最新输出。
- `weka_explorer_stop_clusterer`：请求停止当前原生聚类任务。

支持 `SimpleKMeans`、`FarthestFirst`、`Canopy`、`Cobweb`。SimpleKMeans 可设置 `num_clusters`，范围为 1～100 且不能超过实例数；支持随机种子的聚类器可设置 `seed`。其他选项通过 Weka `OptionHandler` 传递，参数个数和长度有限制。

`weka_explorer_get_state` 现在返回：

- `classifier_running`
- `clusterer_running`
- `preprocess_running`
- `task_running`

任一原生任务运行时，启动另一项分类/聚类任务或修改共享数据会被拒绝。该限制覆盖类别列修改、数据导入、过滤和撤销。

## 实际验证

### 四种聚类算法

使用真实 Explorer 和 Iris 数据，逐一启动四种聚类器并读取结果，均满足：

- 原生任务完成。
- 结果列表存在输出且非空。
- `clusterer_running=false`。
- 没有仅通过类加载或模拟结果代替真实聚类。

SimpleKMeans 通过外部 Java MCP 客户端及随包 STDIO 启动脚本运行，设置 3 个簇、种子 1。`target/native-cluster-result.png` 确认 Cluster 页面存在 `SimpleKMeans` 结果和 `Clustered Instances`，本次 30 条样本分为 3 组，每组 10 条。

宿主输出中仍可能包含 `class` 统计列。桥接不声称类别列一定被移除，类别列、忽略属性和聚类模式遵循宿主面板及其当前设置。

### 长任务取消与互斥

- 使用 100000 条带随机扰动的实例、SimpleKMeans 50 簇和很大的最大迭代数启动长任务。
- 确认任务处于运行状态后，类别列修改和撤销被拒绝。
- 停止请求被接受，随后聚类线程不再运行。
- 在外部 MCP 客户端中，先启动 RandomForest 长分类任务，再请求启动聚类，返回 `Another native Explorer task is already running`。
- 取消分类后，SimpleKMeans 可以正常启动并产生结果。
- 已有分类/聚类结果时执行撤销，返回 `existing_results_may_be_stale`，不将旧评估冒充当前数据结果。

### 回归

原有工作台、分类训练与评估、画像、撤销、模型保存恢复、后台 `weka_load_dataset`、ZIP 安装、类加载、启动脚本版本替换和进程清理均通过回归。

## 边界

- 聚类结果目前返回 Weka 原生文本和模型可用性，未统一为结构化簇分配、质心或评估指标。
- 本版使用宿主当前聚类模式，没有单独验证供应测试集、百分比拆分和 Classes-to-clusters 的全部组合。
- 取消是向 Weka 发出中断请求，第三方算法是否及时响应取决于其实现；本版验证了 SimpleKMeans 长任务取消。
- 尚未接管关联规则、属性选择、任意对话框和可视化，也没有多客户端并发控制。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未实测。
- 发布前仍需完成依赖许可证、版权通知及源码提供义务审查。
