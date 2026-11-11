# Weka MCP workbench

在 Weka Explorer 中提供 **Weka MCP** 标签页，通过独立 Java 子进程运行随包附带的 Weka MCP 1.6.0 服务。数据、实验与导出文件保存在用户工作目录，不需要启动 HTTP 服务。

## 环境要求与安装

- Weka stable 3.8.7；本包不覆盖宿主 Weka 或其他 Explorer 标签页。
- Java 17 或更高版本；GUI 从当前 Java 运行环境启动 MCP 子进程。
- 按 Weka 包管理和 Java 跨平台路径机制设计，不依赖 Bash、PowerShell 或 Python。已实测 macOS 上的 Weka stable 3.8.7 包（运行时版本号为 3.8.8-SNAPSHOT）；Windows、Linux 和更高 Weka 版本仍需在对应环境实测。
- 在 Weka 包管理器中选择本地 `weka-mcp-gui-0.8.0.zip` 安装，然后重启 Explorer。包安装到 `WEKA_HOME/packages/wekaMcp`；未设置 `WEKA_HOME` 时默认是用户主目录下的 `wekafiles/packages/wekaMcp`。

## Agent 原生应用自动化桥接

0.2.0 引入可选桥接，0.3.0 增加原生数据画像与撤销，0.4.0 增加原生聚类与任务互斥，0.5.0 增加原生关联分析，0.6.0 增加原生属性选择，0.7.0 增加原生散点图与 PNG 图像返回，0.8.0 增加结构化结果与数据版本跟踪：MCP Agent 可以读取并操作**当前正在运行的 Explorer**，而不仅是独立后台引擎。默认关闭，需在 **Weka MCP** 标签页点击“启动 Agent 桥接”。“停止”按钮只停止后台 MCP 子进程；“停止 Agent 桥接”才会关闭原生控制通道。

桥接监听随机回环端口，不暴露局域网或 HTTP 服务。连接信息默认写到 `~/.weka-mcp-gui/bridge.json`，含随机端口和访问令牌；POSIX 系统设置为仅当前用户读写。不要分享该文件。每次启用会更换端口和令牌，停止桥接或关闭 Explorer 会删除连接文件。当前同时服务一个 Agent 连接。

### 配置 Agent

macOS/Linux 的 MCP 配置示例（把路径替换为实际 Weka 包目录；若设置了 `WEKA_HOME`，相应调整）：

```json
{
  "mcpServers": {
    "weka-explorer": {
      "command": "/Users/<user>/wekafiles/packages/wekaMcp/bin/bridge-stdio.sh",
      "args": []
    }
  }
}
```

Windows 使用：

```text
C:\Users\<user>\wekafiles\packages\wekaMcp\bin\bridge-stdio.bat
```

连接器本身需要 Java 17+，只加载 GUI JAR 和 Jackson，不加载 Weka 或服务 fat JAR。先启动 Weka 并启用桥接，再让 Agent 连接该 STDIO 命令。不要把 `weka-mcp-java/start.sh` 当作原生桥接入口；它启动的是独立后台服务。

连接信息可用 JVM 属性 `weka.mcp.gui.bridge.file` 改到自定义路径；连接器和 Weka 需使用同一路径。也可将文件路径作为启动脚本唯一参数，或写成 `--bridge-file <path>`。

### 原生工具与后台工具

工具发现返回原有 **43 个后台工具**以及以下 **21 个原生工具**，共 **64 个**：

| 工具 | 作用 |
|---|---|
| `weka_explorer_get_state` | 读取当前数据行列数、类别列、当前标签、分类器和运行状态 |
| `weka_explorer_load_dataset` | 将 ARFF/CSV 加载到 Preprocess，保留撤销点 |
| `weka_explorer_set_class_index` | 设置类别列；`0` 表示最后一列 |
| `weka_explorer_apply_filter` | 对当前数据应用白名单过滤器，保留撤销点 |
| `weka_explorer_describe_dataset` | 返回属性类型、缺失计数、名义值及数值统计，输出数量有上限 |
| `weka_explorer_undo` | 撤销最近一次 Preprocess 数据变更，并等待原生 I/O 完成 |
| `weka_explorer_train_classifier` | 启动 Explorer 的真实交叉验证，不阻塞界面 |
| `weka_explorer_get_results` | 等待并读取原生结果列表中的最新输出 |
| `weka_explorer_stop_classifier` | 请求停止当前原生分类任务 |
| `weka_explorer_train_clusterer` | 启动 Explorer 的真实聚类任务，不阻塞界面 |
| `weka_explorer_get_cluster_results` | 等待并读取原生聚类结果列表中的最新输出 |
| `weka_explorer_stop_clusterer` | 请求停止当前原生聚类任务 |
| `weka_explorer_run_association` | 在 Associate 页面启动 Apriori 或 FPGrowth |
| `weka_explorer_get_association_results` | 等待并读取原生关联分析结果 |
| `weka_explorer_stop_association` | 请求停止当前原生关联任务 |
| `weka_explorer_select_attributes` | 启动原生属性子集评估或排名 |
| `weka_explorer_get_attribute_selection_results` | 等待并读取原生属性选择结果 |
| `weka_explorer_stop_attribute_selection` | 请求停止当前原生属性选择任务 |
| `weka_explorer_visualize` | 为当前数据打开原生散点图，可设置 X/Y 和颜色属性 |
| `weka_explorer_capture_visualization` | 返回原生散点图的 PNG 文件路径及 Base64 图像内容 |
| `weka_explorer_close_visualization` | 关闭桥接自己创建的图窗，不影响其他 Weka 窗口 |

典型调用顺序：加载或读取数据 → 设置类别列 → 预处理 → 启动训练 → `weka_explorer_get_results` 并设置 `wait_seconds`。训练启动后立即返回；不要在未查询结果时假定训练已经完成。

画像统计针对调用开始时的数据快照，默认最多返回 100 个属性、每个名义属性 20 个值，可分别调整到 1000、100。撤销会等待 Weka 原生 I/O 完成；分类、聚类或预处理 I/O 忙时会拒绝执行。撤销只恢复 Preprocess 数据，不删除模型和历史结果，已有结果可能属于旧数据。

原生分类器当前允许 J48、RandomForest、NaiveBayes、Logistic、MultilayerPerceptron、ZeroR；交叉验证折数为 2～50。过滤器当前允许 ReplaceMissingValues、Normalize、Standardize、非监督 Discretize 和监督 Discretize。过滤器选项直接传给对应 Weka `OptionHandler`，错误会返回给 Agent。

原生聚类器当前允许 SimpleKMeans、FarthestFirst、Canopy、Cobweb。`num_clusters` 仅用于 SimpleKMeans，范围为 1～100 且不能超过实例数；支持随机种子的聚类器可设置 `seed`。启动聚类后使用 `weka_explorer_get_cluster_results` 等待结果，使用 `weka_explorer_stop_clusterer` 取消。聚类模式、类别列及忽略属性行为遵循宿主 Cluster 面板，不在桥接层另写一套过滤或评估规则。

`weka_explorer_get_state` 分别返回 `classifier_running`、`clusterer_running`、`associator_running`、`attribute_selection_running`、`preprocess_running` 和 `task_running`。其中任一原生任务运行时，分类、聚类、关联分析、属性选择启动及数据写入会被拒绝，避免共享数据被并发修改。

属性选择支持 `CfsSubsetEval + BestFirst`、`GainRatioAttributeEval + Ranker`、`InfoGainAttributeEval + Ranker`。必须先设置类别列。未指定 `folds` 时使用全量数据评估；指定 2～50 折时使用交叉验证，`seed` 仅用于交叉验证。`num_to_select` 仅用于 Ranker，限制返回的属性数量。评估器和搜索器的选项分别通过 `evaluator_options`、`search_options` 传递。成对切换评估器/搜索器时，仅暂时暂停宿主匹配提示监听，设置完成后恢复，不屏蔽评估异常。

关联分析支持 Apriori、FPGrowth，`num_rules` 可选范围为 1～100，未指定时使用 Weka 默认值。输入必须是名义属性，数值属性先用 Discretize 处理。FPGrowth 在该 Weka 版本中仅接受一元/二元名义属性，桥接会在 **Associate 面板内**将多值名义属性展开为二元属性，保留缺失值，最多扩展到 2000 个属性。此转换不修改 Preprocess 原数据；返回的 `association_data_scope=panel_only_binary_nominal`、`association_attributes` 和 `binary_nominal_expansion` 用于说明实际分析范围。宿主拒绝配置时会返回错误，不再静默无结果。

原生数据导入接受工作区 `data/` 内的相对文件名，或用户信任的本地 ARFF/CSV 文件；外部文件复制进工作区，单个文件限制 100 MB。预处理成功后保留 Weka 撤销点；若处理期间用户改变了数据，插件拒绝覆盖新状态。

**预处理先拟合完整 Explorer 数据集，再做交叉验证，因此不是折内预处理。** 需要严格避免预处理信息泄漏时，应使用后台服务提供的受控预处理/评估流程，而不是把全量预处理结果当作无偏评估。

原生结果保留 Weka 原始输出和模型可用性，分类与聚类同时提供关键结构化指标；关联规则和属性排名尚未全部结构化。其余 43 个工具仍在独立后台进程中执行，不会因为桥接启用就自动操作原生 Classify、Cluster、Associate 或 Visualize 页面。

### 结果指标与数据版本

分类结果的 `evaluation_metrics` 提供正确/错误分类数量及百分比、Kappa、平均绝对误差、均方根误差、相对误差和总实例数；聚类结果提供 `within_cluster_squared_error`。指标来自实际 Weka 汇总文本，原始 `output` 始终保留。

`get_state` 返回当前 `dataset_version`。训练和结果读取提供 `operation_id`、`training_dataset`、`current_dataset`、`dataset_changed_since_training` 与 `result_status`。快照只包含关系名、实例/属性数和类别列，不包含原始数据行。`dataset_version` 是当前会话中用于比较对象身份的标记，不是内容哈希，也不能检测所有原地修改。

状态含义：`RUNNING`（任务进行中）、`COMPLETE`（计算完成）、`STALE_DATASET`（结果属于旧数据）、`CANCELLED`（已取消）、`INCOMPLETE`（尚无完成结果）、`UNTRACKED_HISTORY`（来自未被桥接跟踪的历史结果）。`COMPLETE` 仅表示计算结束，不表示结果仍适用于当前数据。Agent 应结合 `dataset_changed_since_training` 判断是否重新训练。

### 原生散点图

`weka_explorer_visualize` 为当前数据打开 Weka 原生散点图。`x_attribute`、`y_attribute` 默认为前两个属性，`color_attribute` 默认为类别列（无类别列时为最后属性），空字符串可禁用颜色。图像宽度支持 320～1600，高度支持 240～1200。

`weka_explorer_capture_visualization` 必须在打开图窗后调用，返回 PNG 文件路径、尺寸、字节数以及 `type=image`、`mimeType=image/png` 的 Base64 内容。单张图片限制 2 MB，默认保存到工作目录 `data/exports/visualizations/`。它只渲染原生散点图组件，不抓取整个屏幕。

`weka_explorer_close_visualization` 只关闭桥接创建的图窗，不关闭 Explorer 或其他 Weka 图窗。`get_state` 的 `visualization_open` 反映桥接图窗是否打开，它不是后台训练任务。数据变化后需要重新调用可视化以刷新图像；当前不支持分类误差、聚类分配或模型内部结构的专用可视化，也不自动操作任意第三方对话框。

### 安全与协作边界

- 仅监听回环地址，并要求随机令牌；不要把令牌文件放进共享目录、聊天记录或版本库。
- 原生训练期间避免人工同时更改分类器或数据。预处理提交前会检查数据是否变化，但这不替代用户与 Agent 的操作协调。
- 关闭桥接或 Explorer 会请求停止原生训练并清理连接；强制结束 Weka 进程后，可能遗留已失效的连接文件，重新启用时会替换。
- 这是本机自动化接口，不是沙箱。只向受信任的本地 Agent 开放，不要把回环端口转发到远程机器。

## 使用流程

1. 在 **Preprocess** 打开 ARFF/CSV 后切到 **Weka MCP**，或在本标签页点击“选择 ARFF/CSV…”。点击“加载与拆分”会读取数据、生成画像，并以固定种子 1 做 80%/20% 拆分。
2. 设置类别索引（`0` 表示最后一列，其他值为从 1 开始的列号）、分类器类名和 Weka 选项，点击“训练与评估”。
3. “预测”使用当前模型的测试集；对于从 Explorer 传入的数据，使用第一条非类别属性记录做单条预测。实验会保存训练配置和评估结果。
4. “聚类”使用 SimpleKMeans、3 个簇并排除类别列，返回聚类评估。首版的算法和参数在界面固定，其他算法可通过底层 MCP 服务调用。
5. 记录训练后显示的实验 ID。“保存实验”显式保存模型；“恢复模型”按 ID 恢复，存在测试集时重新评估。“导出模型”支持 `weka_bundle`、原始模型和 PMML。导出路径显示在结果区域。

“恢复模型”使用 MCP 生成的实验 ID，而不是 Weka Explorer 的实验结果编号。恢复服务会话中的模型后如需测试集，请重新加载和拆分数据。原生模型导入应在“信任该文件”的前提下使用，不要加载来源不明的 Java 序列化文件。

## 文件位置

默认工作目录为 `~/.weka-mcp-gui`：

- `data/`：数据集副本、从 Explorer 写出的 ARFF、导出文件。
- `experiments/`：可跨进程恢复的模型实验。
- `config.properties`：每次启动生成的隔离 MCP 配置。

高级用户可在启动 Weka 前设置 `weka.mcp.gui.workspace` 覆盖工作目录，`weka.mcp.gui.package` 覆盖包目录。它们属于 JVM 系统属性，不是 MCP 配置项。

停止服务或关闭 Explorer 会关闭子进程。首版不占用 Explorer 原生结果缓冲区，不提供 AutoML、可视化、关联规则等全部 43 个 MCP 工具的专属界面；它提供的是核心分析工作台，而非完整替换 Explorer。

## 构建

在同时包含 `weka-mcp-java/` 和 `weka-gui-plugin/` 的父目录运行：

```sh
cd weka-mcp-java && mvn -B -ntp clean verify
cd ../weka-gui-plugin && mvn -B -ntp clean verify
```

生成 `weka-gui-plugin/target/weka-mcp-gui-0.8.0.zip`。GUI 测试使用隔离的 Weka 主目录，不会安装到用户真实 Weka 配置。集成测试需要可用的图形桌面，会短暂显示真实 Explorer；在无显示器的 CI 环境需配置图形会话，不能仅用 `-Djava.awt.headless=true` 运行。

本包包含 Weka、MCP Java SDK、Jackson、Logback 及其传递依赖。许可证、第三方通知与对应源码获取方式见 `licenses/README.md` 和 `THIRD_PARTY_NOTICES.md`；各组件保留自己的许可证。
