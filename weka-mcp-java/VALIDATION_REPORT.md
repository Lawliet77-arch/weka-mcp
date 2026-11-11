

## 13. 1.6.0 本地常规 Weka 工作流扩展验证

### 验证结论

本轮针对算法与工具广度、搜索与评估、模型互操作三个方向完成扩展。服务升级为 **1.6.0**，MCP 工具由 28 个增加到 **43 个**，白名单由 12 个分类器、7 个过滤器、3 个聚类器扩展为 **36 个分类器、18 个过滤器、6 个聚类器、2 个关联算法**，共 **62 个受控条目**。

最终在本机 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16 环境执行：

```bash
mvn -B -ntp clean verify
```

结果为 **BUILD SUCCESS**：

| 测试类别 | 数量 | 失败 | 错误 | 跳过 |
|---|---:|---:|---:|---:|
| 单元测试 | 48 | 0 | 0 | 0 |
| 真实 MCP 子进程测试 | 3 | 0 | 0 | 0 |
| 合计 | 51 | 0 | 0 | 0 |

相较 1.5.0 新增 15 个单元测试。真实 MCP 测试扩展了工具发现、数据拆分与导出、推荐、阈值与统计比较、嵌套自定义候选、GUI 模型导入、PMML、续训及跨 JVM 恢复，而不是仅验证类能否加载。

### 算法与工具广度验证

新增的 24 个分类算法均完成适合其能力的真实训练：

- 19 个分类算法覆盖 OneR、DecisionStump、JRip、PART、DecisionTable、BayesNet、NaiveBayesMultinomial、RandomTree、REPTree、HoeffdingTree、SimpleLogistic、MultilayerPerceptron、KStar、LWL、Bagging、LogitBoost、RandomCommittee、RandomSubSpace 和 NaiveBayesMultinomialUpdateable。
- M5Rules、M5P、AdditiveRegression、RegressionByDiscretization 完成回归训练与评估。
- VotedPerceptron 完成二分类训练。
- FarthestFirst、Canopy、Cobweb 完成聚类训练。

`weka_list_algorithms` 实测返回 62 个条目及选项、能力、最小实例数和增量标记。无 `OptionHandler` 的 DecisionStump 使用默认配置；有选项算法仍执行严格校验。MultilayerPerceptron 被正式支持后，旧的拒绝用例改为未列入白名单的 DL4J，未通过放宽类加载来消除测试失败。

本地 Weka 3.8.7 没有 XMeans 或 `ChiSquaredAttributeEval`，因此没有声明支持这些类；属性选择使用实际存在的 GainRatio、SymmetricalUncert、OneR、ReliefF、InfoGain 和 Cfs，并逐一返回排名与分数。

### 数据准备与可视化验证

- `weka_split_dataset` 在固定种子下得到 24 条训练实例、6 条测试实例，并完成训练后的独立评估。
- Normalize 预览返回数值行和 `fit_scope=full_dataset`、`fold_safe=false`；监督离散化使用 Weka 实际支持的选项，未设置类别索引时明确拒绝。
- 数值直方图 bin 计数总和为 30，三个 Iris 类别各有 10 条；散点采样保留类别标签。
- 四个数值属性生成 6 组散点配对图，采样上限生效。
- PSI 漂移输出包含数值和名义分布；属性按名称定位，不依赖 Weka `List.indexOf` 的对象身份语义。
- ARFF 和 CSV 导出后均重新加载为 30 条、5 个属性。默认不覆盖和路径穿越拒绝均通过。
- FPGrowth 对多值名义属性进行二值名义编码并返回规则；数值离散化、二值扩展上限和结果截断均有控制。

### 搜索与评估验证

`weka_evaluate` 的 `repeats` 返回每次结果及聚合值。两次三折评估的混淆矩阵总数为 60，逐类 support 为 20，准确率等于两次结果的均值；重复留出评估也通过。

`weka_compare_models` 在共享折上比较 ZeroR 与其他模型，2 次三折得到 6 组配对结果，返回校正重采样配对 t 检验、均值差、标准误、置信区间和精确双尾 p 值。相同种子的结果可复现，回归 RMSE 比较也通过。

t 分布计算以理论值验证：

- `df=1, t=1` 的双尾概率为 0.5。
- `df=2, t=2` 的双尾概率为 `1 - 2/√6 ≈ 0.183503419`。
- `df=10, t=2.228138851986273` 的双尾概率为 0.05。

实现使用独立的正则化不完全 Beta 连分式及 Weka 已核对的对数 Gamma 值，不采用固定 `2.045` 临界值。测试期间曾把第二个理论值误写为 `1/3`，随后按公式修正测试预期；没有把这个测试错误归因于 Weka 库。

嵌套逐次减半通过 3 个外层折、2 个内层折：每个外层训练折独立完成 25%、50%、100% 资源淘汰，测试折只用于评分。实验包含外层 winners、折内候选、阶段及完成标记，模型恢复后的预测一致。预算耗尽时可回退到 `outer_training_fold`，并明确返回 `partial=true`、`nested_cv_complete=false` 和 `final_model_status=NOT_TRAINED`，不写入未完成实验。

自定义 `candidates` 已验证：RandomTree 带受控选项、NaiveBayes 与 ZeroR 基线按 3→2→1 淘汰，支持嵌套逐次减半、随机顺序复现和 `include_baseline`。`inner_folds` 与普通 `folds` 已分离，避免混用内外层协议。

二分类阈值测试精确计算 `2 × cost_fn = 6` 的成本；另一个 NaiveBayes 用例返回 21 个阈值、6 个评分实例和 0 最小成本。类别概率、边际分布、非法阈值拒绝均通过。

### 模型互操作验证

- J48 的 `weka_bundle` 依次包含分类器与训练表头，导入后预测、独立评估及持久化恢复一致。
- RandomForest、Logistic 的原生模型导入后预测与导出前一致；内部基类、过滤器及必要依赖按实际序列化结构补充校验。
- SimpleKMeans 原生聚类器导入后簇分配一致，支持实验恢复。
- Logistic 的 PMML 输出通过 XML 解析验证根元素和模型节点；J48 未实现 `PMMLProducer`，返回明确错误，不生成伪 PMML。
- `weka_update_model` 从两条新 ARFF 实例续训，预测概率与直接 Weka `NaiveBayesUpdateable` 逐项一致，实验记录 `UPDATE_SOURCE` 和父实验 ID。
- 配额不足导致续训保存失败时，原会话模型的预测保持不变，验证克隆更新和失败回滚。
- 默认不覆盖、原子写入、输出路径限制、不可信导入拒绝、错误文件拒绝及表头校验均通过。

模型导入不是对任意 Java 序列化文件的安全承诺。调用者必须显式传入 `trusted=true`；只允许白名单模型、继承结构及明确的内部依赖。导入模型保留原配置并标记 `retrainable=false`，用于预测、直接评估和受控续训，不直接用外部选项重新训练。

### 与公开同类项目的定位比较

本轮对比参考了 [iamademar/weka-mcp](https://github.com/iamademar/weka-mcp) 的 README 和训练、实验、模型服务源码。其 README 当前列出 46 个工具，覆盖更宽泛的动态算法调用、数据可视化和 HTTP/Azure 部署。

本版本的差异不再是单纯工具数量，而是以下本地工作流能力：

- 受控算法能力发现、类型兼容推荐、文本预处理、数据漂移和可复现拆分。
- 嵌套逐次减半、外层 winners 与完成度血缘、自定义候选和总预算控制。
- 精确双尾 p 值的校正配对比较，而非固定临界值近似。
- 同时提供实验封装、带训练表头的 GUI 模型包、原生模型、PMML、增量续训与失败回滚。

这使本地常规分析闭环更完整，但不能据此宣称全面覆盖 Weka 的全部算法或所有生产场景。对方更宽的动态算法调用和远程部署仍有优势；本轮未运行对方的测试套件，也没有做性能基准、第三方安全审计或社区采纳比较。

### 本轮发现并修复的实现问题

1. 根据真实 Weka 类型补齐 DecisionStump 包名、无 `OptionHandler` 模型支持、监督离散化选项及属性选择器基类。
2. 根据实际序列化结构补齐分类器/聚类器基类、内部过滤器与继承类型，未简单开放全部 `weka.*`。
3. 修复嵌套搜索内外层折数混用、全量选择尚未开始时的部分状态回退，以及外层候选血缘保留。
4. 修复数据漂移比较中依赖 `Instances.indexOf` 的对象身份查找问题。
5. 修正测试中对不同算法概率分布的错误横向比较，改为逐模型验证导出前后一致性。
6. PMML 验证改用真实 XML 解析，而不是依赖猜测的标签文本。

### 最终产物

```text
target/weka-mcp-java-1.6.0.jar
```

- 大小：**35,048,050 字节**。
- SHA-256：`a6178134cef3699ead4a683290132fee08cc0ed5cf185cfecff93983177e980d`。
- `pom.xml`、服务版本、启动脚本、集成测试和 CI 产物路径已同步为 1.6.0；`start.bat` 仍为 CRLF。
- 全部 43 个工具的调用方式、示例和边界已更新到 `README.md`。

### 仍需保留的限制

1. 本项目是本地单会话 STDIO 服务，不包含 GUI、HTTP、认证、多租户或 Weka GUI 内部共享状态。
2. 受控条目不是 Weka 的全部算法；没有开放任意类加载。普通数据加载仍受内存限制，流式 IBk 也保存训练实例。
3. 搜索使用有界参数空间和固定资源阶段，不是完整贝叶斯优化或 Hyperband。预算是软取消，不保证所有 Weka 算法立即停止。
4. 导入只处理可信 Weka 模型。PMML 只覆盖 `PMMLProducer`，不提供通用 PMML 导入或任意预处理管线的 PMML 导出。
5. 测试以小型真实数据和工作流为主，没有宣称大数据、长期运行、高并发、性能基准或严格生产隔离已验证。
6. **Windows 仍未实际运行通过。** CI 配置和 Windows 分支存在，但本机验证不能代替 Windows 验收。
7. macOS ARM64 的 netlib 警告通过纯 Java 回退处理，Shade 重复资源警告仍存在。
8. 尚无公开仓库、社区采纳数据或第三方安全审计。`LICENSE` 和 fat jar 依赖分发审查仍是发布前待办。
