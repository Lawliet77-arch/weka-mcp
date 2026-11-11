# weka-mcp-java

纯 Java、单进程、STDIO 传输的 Weka MCP 服务。Weka 与 MCP Java SDK 运行在同一 JVM 中，不包含 HTTP 服务、GUI、认证或多租户代码。

当前版本 **1.6.0**，提供 **43 个工具**。覆盖数据准备、探索分析、训练、重复评估、统计比较、增量续训、实验持久化，以及 Weka GUI/PMML 模型互操作。

## 技术基线

- Java 17
- MCP Java SDK 2.1.0
- Weka stable 3.8.7
- Jackson 3.1.4
- Maven 3.9.x

## 构建与启动

```bash
mvn clean verify
```

产物：`target/weka-mcp-java-1.6.0.jar`。这是包含 Weka 的可执行 fat jar，不需要另外安装 Weka 或 Node.js。

将 ARFF/CSV 放入 `data` 目录，在 MCP 客户端中配置：

```json
{
  "mcpServers": {
    "weka": {
      "command": "/absolute/path/to/weka-mcp-java/start.sh"
    }
  }
}
```

Windows 使用 `start.bat`。脚本设置 2 GB 最大堆、UTF-8、无头模式和配置文件位置，可通过 `WEKA_MCP_HEAP` 调整堆大小。直接双击 jar 不会打开 GUI；它等待 STDIO 上的 MCP 请求。

独立部署需要以下文件位于同一目录：

```text
weka-mcp-java-1.6.0.jar
start.sh / start.bat
config.properties
weka-whitelist.yaml
automl-search.yaml
```

开发目录中的脚本也会自动使用 `target/weka-mcp-java-1.6.0.jar`。

## Windows 适配与验证

`start.bat` 优先使用有效的 `JAVA_HOME`，否则回退到 PATH 中的 `java.exe`；jar 缺失时错误写入 stderr，Java 退出码传回客户端。`.gitattributes` 固定 BAT 使用 CRLF、shell 脚本使用 LF。

Windows 的 `config.properties` 建议用正斜杠写绝对路径：

```properties
dataRoot=C:/weka-data
whitelistPath=C:/weka-mcp/weka-whitelist.yaml
```

源码仓库根目录的 `.github/workflows/verify.yml` 在 Ubuntu、macOS、Windows 的 JDK 17 上对两个模块执行 `mvn -B -ntp clean verify`，并上传 GUI ZIP、服务 JAR 和测试报告。后台测试覆盖中文和空格的部署目录、CRLF ARFF、加载后重命名、stdout 协议纯净性以及 stdin EOF 退出；GUI 工作流运行真实 Explorer 测试。

**Windows BAT 和原生桥接以该工作流的实际运行结果为准，不能将本机 macOS 验证结果称为 Windows 已验证。**

## 工具清单

| 工具 | 功能 |
|---|---|
| `weka_load_dataset` | 加载数据根目录中的 ARFF/CSV |
| `weka_describe_dataset` | 属性名、类型、名义值和类索引 |
| `weka_list_resources` | 枚举会话数据集、模型和聚类器 |
| `weka_list_algorithms` | 查询受控算法、选项、能力及增量支持 |
| `weka_recommend` | 按类型兼容和复杂度推荐候选，不声称性能排名 |
| `weka_list_filters` | 列出过滤器及允许的选项 |
| `weka_apply_filter` | 应用过滤器并注册新数据集 |
| `weka_split_dataset` | 生成可复现的分层训练集和测试集 |
| `weka_transform_preview` | 预览转换结果，明确标记全量拟合和非折内语义 |
| `weka_histogram` | 有界数值/名义直方图 |
| `weka_scatter` | 有界散点采样及类别标签 |
| `weka_scatter_matrix` | 最多 5 个数值属性的配对散点图 |
| `weka_compare_datasets` | 表头比较、缺失统计和 PSI 数据漂移 |
| `weka_export_dataset` | 原子导出 ARFF/CSV，默认不覆盖 |
| `weka_profile_dataset` | 缺失、分布、重复行、类别失衡和相关分析 |
| `weka_learning_curve` | 固定外层测试集上的学习曲线 |
| `weka_model_diagnostics` | PR/ROC、校准、残差及错误实例 |
| `weka_threshold_analysis` | 二分类阈值、成本曲线和边际分布 |
| `weka_select_attributes` | InfoGain、Cfs、GainRatio、SymmetricalUncert、OneR、ReliefF |
| `weka_associate` | Apriori/FPGrowth，支持受控离散化和二值化 |
| `weka_explain_model` | J48 文本树和 DOT |
| `weka_train_classifier` | 训练分类器或回归器，支持折内预处理和持久化 |
| `weka_train_stream` | 从 ARFF/CSV 逐条训练五种增量算法 |
| `weka_auto_classify` | 搜索、无偏评估、最终训练及实验保存 |
| `weka_evaluate` | 单次或重复交叉验证/留出评估 |
| `weka_compare_models` | 共享折重复比较和校正配对 t 检验 |
| `weka_evaluate_model` | 在提供的标注测试集上直接评估已有模型 |
| `weka_list_experiments` | 分页列出持久化实验 |
| `weka_get_experiment` | 实验配置、血缘、搜索阶段、来源和指标 |
| `weka_save_model` | 保存会话分类器，返回 `experiment_id` |
| `weka_load_experiment_model` | 跨进程恢复分类模型 |
| `weka_save_clusterer` | 保存聚类器、标签移除位置和簇分配 |
| `weka_load_clusterer` | 跨进程恢复聚类器 |
| `weka_export_model` | 导出 Weka GUI 模型包、原生模型或 PMML |
| `weka_import_model` | 导入可信 Weka 模型并验证类型和表头 |
| `weka_update_model` | 从会话数据集或 ARFF/CSV 增量续训 |
| `weka_compare_experiments` | 按验证或测试分数比较实验 |
| `weka_delete_experiment` | 删除实验 |
| `weka_predict` | 对兼容数据集或内联实例预测 |
| `weka_cluster` | 训练聚类器，支持标签移除和持久化 |
| `weka_cluster_predict` | 为新实例分配簇 |
| `weka_cluster_evaluate` | 簇大小、类别映射、似然及有界 Silhouette |
| `weka_release` | 释放会话资源 |

`class_index` 从 1 开始。训练时未指定则复用数据集类索引；若仍未指定，默认最后一个属性。

## 算法与分析覆盖

`weka-whitelist.yaml` 当前包含 **36 个分类器、18 个过滤器、6 个聚类器、2 个关联算法**，共 62 个受控条目。覆盖树、规则、贝叶斯、Logistic、SMO/SMOreg、线性模型、神经网络、懒惰学习、集成方法、回归和文本向量化。

`weka_list_algorithms` 返回每个条目的选项规则、支持的属性/类别类型、最小实例数和增量能力。无 `OptionHandler` 的算法允许使用默认配置，但不能传入任意选项。`weka_recommend` 只做兼容性与复杂度启发式筛选，必须再用 `weka_auto_classify` 或独立评估验证。

当前 Weka 3.8.7 中没有 XMeans 或 `ChiSquaredAttributeEval`，因此不虚构支持；属性选择提供实际可用的 `symmetrical_uncert` 等方法。

## 本地数据工作流

1. `weka_load_dataset` 加载文件。
2. `weka_profile_dataset`、`weka_histogram`、`weka_scatter` 或 `weka_scatter_matrix` 检查数据。
3. `weka_transform_preview` 验证过滤效果；`weka_apply_filter` 注册转换后的数据。
4. `weka_split_dataset` 创建固定种子的分层训练集和测试集。
5. 训练、独立评估、阈值分析、模型比较和聚类评估。
6. `weka_export_dataset` 保存处理结果，`weka_export_model` 导出模型。

转换预览和手动全量过滤不自动保证折内无泄漏。需要训练/评估预处理时，应使用 `auto_preprocess` 或 `weka_auto_classify`。预览会明确返回 `fit_scope=full_dataset` 和 `fold_safe=false`。

PSI 是描述性漂移统计，输出低、中、高参考区间，不是显著性检验。散点、Silhouette、直方图和规则数量均有上限。

## 搜索与评估

### AutoML

`weka_auto_classify` 默认比较 5 个分类候选或 3 个回归候选并加入 ZeroR 基线。支持：

- `search=defaults|grid|random|halving`。
- `protocol=holdout|nested_cv|legacy_cv`。
- `optimize=accuracy|weighted_f1|rmse`。
- `maxCandidates` 和覆盖搜索、测试、最终训练的总 `budgetSeconds`。
- `auto_preprocess=true`：在训练折内拟合缺失值处理、文本向量化和尺度处理。

`protocol=holdout` 先隔离外层测试集，只用训练部分筛选候选；`nested_cv` 在外层训练折内调参，测试折只用于评分；`legacy_cv` 用于复现旧评估语义。

逐次减半在 25%、50%、100% 数据量上淘汰候选，例如 6→3→2。`nested_cv` 下，每个外层折都独立进行资源淘汰，记录 `nested_outer_folds`、外层 winners、折内候选和完成度。`selection_source=full_data` 表示完成全量选择；预算过早耗尽时可回退到 `outer_training_fold`。未完成最终模型时返回 `final_model_status=NOT_TRAINED`，不会写入不完整实验。

可直接传候选配置，无需编辑 YAML：

```json
{
  "dataset_id": "数据集 UUID",
  "classifier_class": "auto",
  "search": "halving",
  "protocol": "nested_cv",
  "outer_folds": 3,
  "inner_folds": 2,
  "candidates": [
    {"classifier_class": "weka.classifiers.trees.RandomTree", "options": ["-K", "2"]},
    {"classifier_class": "weka.classifiers.bayes.NaiveBayes"}
  ],
  "include_baseline": true,
  "budgetSeconds": 120
}
```

`automl-search.yaml` 也包含扩展候选。所有候选都复用白名单、折内预处理、固定种子、预算和实验血缘。

### 重复评估与统计比较

`weka_evaluate` 新增 `repeats`（1–20），返回每次指标以及 `mean_of_repeats` 聚合；混淆矩阵和 support 累加，其他指标按重复次数求均值。

`weka_compare_models` 接受 2–8 个 `model_ids`，所有模型使用同一组共享训练/测试折，预处理在每折重新拟合。支持 accuracy、weighted F1、RMSE，返回：

- 每个模型的逐折分数、均值、标准差和标准误。
- 校正重采样配对 t 检验的均值差、标准误、自由度、t 值、精确双尾 p 值和置信区间。
- 以第一项模型为基线的胜负或 tie，以及只包含完整折的 `partial` 状态。

校正方差使用 `1/N + 1/(folds-1)`，不采用固定近似临界值。导入模型不能直接用于需要重新训练的重复比较，应使用 `weka_evaluate_model` 做直接评估。

`weka_threshold_analysis` 仅用于二分类，可指定 `positive_class`、阈值列表、`cost_fp`、`cost_fn` 和边际直方图。它直接复用已训练模型和预处理，不重新训练，输出最佳阈值、混淆矩阵、F1、误报率、成本和边际分布。

### 聚类与关联分析

`weka_cluster_evaluate` 返回簇大小、类别到簇的映射、密度模型对数似然和采样 Silhouette。`max_samples` 限制二次距离计算规模。

`weka_associate` 支持 Apriori 和 FPGrowth。`auto_prepare=true` 时对数值属性离散化，并为 FPGrowth 将多值名义属性转换为二值名义属性；返回转换后的属性数，不替换会话数据集。规则数、置信度、支持度、Lift 和截断标志均有返回。

## 模型互操作与续训

### 持久化

实验默认位于 `data/experiments/<experiment-id>/`，包含 `experiment.json` 和 `model.model`。普通分类器、AutoML、聚类器还会保存 `dataset.arff`；流式实验只保存模型、源文件相对路径、哈希和元数据。

使用 `persist=true` 或 `weka_save_model`、`weka_save_clusterer` 显式保存。实验保存算法选项、预处理、表头、指标、搜索阶段、父实验和更新来源。`weka_load_experiment_model` 和 `weka_load_clusterer` 跨进程恢复，普通会话 UUID 不跨进程复用。

### 导出

`weka_export_model` 支持：

- `weka_bundle`：依次写入模型和训练表头，采用 Weka GUI 读取的格式。
- `raw_model`：仅写入 Weka `Classifier` 或 `Clusterer` 对象。
- `pmml`：通过 `PMMLProducer.toPMML(header)` 导出；当前已验证 Logistic，其他未实现该接口的模型会明确拒绝。

输出路径必须在数据根目录内，使用临时文件和原子替换；默认不覆盖，需显式 `overwrite=true`。GUI 模型文件和 PMML 都是互操作格式，不等同于本服务实验目录中的封装 `model.model`。

### 导入与续训

`weka_import_model` 只处理数据根目录内的可信 Weka 模型，必须显式传 `trusted=true`。它使用反序列化类型、深度、引用和字节限制，检查允许的模型及内部依赖、预处理过滤器和训练表头。Java 反序列化不能视为对任意不可信文件安全，因此不要导入未知来源文件。

导入模型保留原配置并标记 `retrainable=false`，避免将外部选项直接用于重新训练。可继续 `weka_predict`、`weka_evaluate_model` 和 `weka_threshold_analysis`，也可持久化恢复。

`weka_update_model` 支持直接实现 `UpdateableClassifier` 的模型，输入可为 `dataset_id` 或 ARFF/CSV `path`。服务先克隆模型，再应用更新，成功写入新实验后替换会话对象；持久化失败不会污染原模型。新实验保存父实验 ID 和更新数据来源。

### 五种流式算法

- `weka.classifiers.bayes.NaiveBayesUpdateable`
- `weka.classifiers.bayes.NaiveBayesMultinomialUpdateable`
- `weka.classifiers.lazy.IBk`
- `weka.classifiers.functions.SGD`
- `weka.classifiers.trees.HoeffdingTree`

`weka_train_stream` 使用 `getStructure()` 和 `getNextInstance()` 逐条读取，不调用批量 `getDataSet()`：

```json
{
  "path": "iris.arff",
  "classifier_class": "weka.classifiers.bayes.NaiveBayesUpdateable",
  "class_index": 5,
  "evaluation": "holdout",
  "test_every": 5,
  "max_instances": 100000,
  "max_test_instances": 1000,
  "budgetSeconds": 120,
  "persist": true
}
```

默认每第 5 条留出为独立测试集；`prequential` 则先测后学。测试数量、读取数量、属性数和时间均受限，不累积全部预测对象。IBk 等模型仍会保存训练实例，不是恒定内存算法。

## 配置

主要配置位于 `config.properties`，均可用环境变量覆盖：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `dataRoot` | `data` | 数据及导出的根目录 |
| `searchSpacePath` | `automl-search.yaml` | 候选参数空间 |
| `experimentRoot` | `data/experiments` | 实验目录 |
| `maxExperiments` | 100 | 最大实验数 |
| `maxExperimentDiskBytes` | 536870912 | 实验磁盘配额 |
| `maxModelBytes` | 67108864 | 模型文件上限 |
| `autoMlBudgetSeconds` | 300 | AutoML 总软预算 |
| `maxDatasets` | 20 | 会话数据集容量 |
| `maxModels` | 50 | 会话分类模型容量 |
| `maxClusterers` | 20 | 会话聚类器容量 |
| `maxInstances` | 100000 | 普通数据集或流式读取实例上限 |
| `maxAttributes` | 10000 | 属性数上限 |
| `maxFileBytes` | 104857600 | 输入文件大小上限 |
| `maxOptionsLength` | 20 | 选项 token 数上限 |
| `maxInstanceJsonBytes` | 16384 | 内联实例字节上限 |
| `maxPredictionRows` | 1000 | 批量预测/簇分配返回行数上限 |
| `maxOutputBytes` | 1048576 | 工具 JSON 输出上限 |
| `trainTimeoutSeconds` | 120 | 训练、评估和聚类响应超时 |
| `loadTimeoutSeconds` | 30 | 加载、过滤和预测响应超时 |
| `maxConcurrentTasks` | 2 | 最大并发任务数 |
| `maxQueuedTasks` | 8 | 等待队列上限；0 表示不排队 |

可通过 `WEKA_MCP_CONFIG` 或 `-Dweka.mcp.config=...` 指定配置文件。环境变量示例：

```bash
export WEKA_MCP_DATA_ROOT=/absolute/path/to/data
export WEKA_MCP_MAX_DATASETS=10
```

## 安全与错误

白名单外类、未批准选项、重复选项、越界值、非 UUID、路径穿越、符号链接逃逸、非有限数值和超限输出都会被拒绝。stdout 只承载 JSON-RPC；日志写 stderr，不记录完整数据集或 `instance_json` 原文。

成功结果的 JSON 在 `content[0].text`，错误示例：

```json
{
  "content": [{"type": "text", "text": "{\"error\":\"可读错误信息\"}"}],
  "isError": true
}
```

## 当前边界

- 本服务仍是本地单会话 STDIO 服务，不提供 GUI、HTTP、多租户、认证或 Weka GUI 内部共享状态。
- 数据集默认只存在于会话中；模型和聚类器需要显式持久化。`persist=false` 的流式模型没有训练数据或源文件引用，不能随后通过 `weka_save_model` 保存。
- 只开放白名单算法及其必要内部依赖，不覆盖 Weka 的全部算法和包；不提供任意类加载或任意外部 Java 模型导入。
- 导入限制只处理可信 Weka 模型，不能保证任意 Java 序列化输入安全；PMML 仅覆盖实现 `PMMLProducer` 的模型，不提供通用 PMML 导入。
- 搜索空间有界，不包含贝叶斯优化。逐次减半使用固定的 25%、50%、100% 资源阶段，不是完整 Hyperband 实现。
- 自动预处理复用管线，但不会自动对齐不同属性名、顺序或类型。手动全量过滤后再交叉验证仍可能泄漏。
- 预算与超时是软取消，Weka 不保证立即响应线程中断，不是进程级硬隔离。普通数据加载仍受 JVM 内存限制。
- 流式训练不自动进行全量预处理；周期留出不打乱输入，预序列指标也不是独立测试成绩。
- 尚无第三方安全审计或社区采纳数据。项目采用 GPL-3.0-or-later；fat jar 内携带 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和第三方原始许可证/NOTICE。各依赖保留自己的许可证，对应源码获取方式见根目录通知文件。
- macOS ARM64 的 netlib 原生库警告会回退到纯 Java；Shade 的重复资源警告仍存在。Windows 分支必须实际 CI 通过后才算验证。

## 测试

```bash
mvn test          # 单元测试
mvn clean verify  # 单元测试 + fat jar STDIO 端到端测试
```

当前验证包含 **48 个单元测试和 3 个真实 MCP 子进程测试，共 51 个**，详见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。

覆盖全部新增算法、五种流式算法、重复评估、精确 t 分布尾概率、嵌套逐次减半、自定义候选、PMML、模型导入导出、续训回滚、数据预览、统计可视化、漂移比较、路径与容量限制，以及跨 JVM 模型和聚类器恢复。
