# Weka MCP GUI 0.1.0 验证报告

## 结论

已完成独立服务与 GUI 工程的目录分离，并实现了可安装的 **Weka Explorer「Weka MCP」标签页**。本机最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**，共 **7 个测试通过，0 失败、0 错误、0 跳过**。

这是核心分析工作台首版，不是对全部 43 个 MCP 工具的图形化封装，也未完成 Windows、其他 Weka 发行版或发布级许可证审计。

## 目录与交付物

- `weka-mcp-java/`：原纯 Java STDIO 服务 1.6.0，含源码、测试、脚本、配置、文档和部署目录。
- `weka-mcp-java/dist/weka-mcp-java-1.6.0/`：不接入 GUI 的独立服务部署目录。
- `weka-gui-plugin/`：GUI 源码、测试、安装元数据和验证报告。
- `weka-gui-plugin/target/weka-mcp-gui-0.1.0.zip`：通过 Weka 包管理器离线安装的 ZIP。

GUI 使用宿主的 Weka 类；`tools/` 内的 MCP fat JAR 只作为独立子进程运行，不加载为 GUI 库。配置与模型实验独立于 Explorer 的结果缓冲区。

## 验证环境

- macOS ARM64，Temurin JDK 17.0.21+9，Maven 3.9.16。
- Maven 坐标 `nz.ac.waikato.cms.weka:weka-stable:3.8.7`。
- 该 JAR 的 `weka/core/version.txt` 和原生窗口实际显示 **3.8.8-SNAPSHOT**。因此安装约束采用 `weka (>=3.8.7)`，而不是会导致包被跳过的精确 `=3.8.7`。
- 测试使用 `target/test-weka-home`、离线包管理模式及隔离工作目录，不写入用户真实的 `~/wekafiles` 或 `~/.weka-mcp-gui`。
- 本机未安装可确认的 Weka 桌面发行 App，验收使用上述参考 Weka JAR，不能据此声称已验证其他发行版。

## 构建与安装验证

执行：

```sh
cd weka-mcp-java && mvn -B -ntp clean verify
cd ../weka-gui-plugin && mvn -B -ntp clean verify
```

迁移后独立服务已通过 48 个单元测试和 3 个真实 MCP 子进程测试；该服务原有验证细节见 `weka-mcp-java/VALIDATION_REPORT.md`。

GUI 最终结果：

| 测试 | 数量 | 覆盖内容 |
|---|---:|---|
| 单元测试 | 4 | 真实 STDIO 初始化及工具结果解析、模型训练/预测/聚类、实验恢复、模型导入导出、增量续训、控制器启停和 EDT 回调、Explorer 接口 |
| 集成测试 | 2 | ZIP 安装和包管理器加载、可见 Explorer 的完整界面操作链及子进程退出 |
| 合计 | 7 | 全部通过 |

`PackageInstallationIT` 调用真实 `WekaPackageManager.installPackageFromArchive`，确认安装后的服务 JAR、运行库、白名单与标签配置存在；还断言 `loadCheck`、`hasBeenLoaded`、包类加载器及 Explorer 标签实例，避免仅因测试 classpath 中有源码而产生假阳性。

使用 Maven 解析出的宿主依赖启动独立 Explorer（不把 GUI 编译目录加入 classpath），实际观察到原生 Explorer 的 **Weka MCP** 标签。GUIChooser 已正常退出，所有测试和验收窗口均关闭。

## 可见界面操作链

`McpPanelWorkflowIT` 创建可见的真实 Explorer，通过 Preprocess API 传入 30 条 Iris 测试数据，选中包管理器注册的 MCP 标签页，并用 Swing `doClick()` 调用真实按钮监听器。它不是仅实例化一个孤立 JPanel，也不是模拟 MCP 返回。

已验证：

1. 启动 MCP 子进程并完成 JSON-RPC 初始化。
2. 从 Explorer 接收数据，写出 ARFF、生成画像并完成固定种子的 80%/20% 拆分。
3. 使用 J48 训练并在独立测试集评估，结果包含准确率。
4. 单实例预测得到 `Iris-setosa`。
5. SimpleKMeans 聚类并返回 3 个簇及评估。
6. 保存实验，读取 MCP 实验 ID，导出包含训练表头的 GUI 兼容模型包；文件确实存在。
7. 停止并重新启动服务，按实验 ID 恢复模型。
8. 关闭 Explorer 后匹配的 MCP 子进程数为 0。
9. 生成 `target/gui-workflow.png`，检查按钮、数据接收、实验 ID、导出路径和状态文本的显示。

这是自动驱动的可见 GUI 验证，不声称已经完成逐个人工鼠标操作。文件选择器、原生 PMML 选择框和不同窗口尺寸还需补充人工/跨平台界面测试；PMML 能力另由真实 MCP 子进程测试覆盖。

## 本轮修复的问题

- 修正 Jackson 3 对缺失 `isError` 的转换，以及 MCP 文本内容中的 JSON 结果解析。
- 修正 Java 可执行文件定位中的变量遮蔽、泛型转换和同步/异步调用接口。
- 修正 Maven 本地服务 JAR 的打包方式，不引入额外依赖。
- 修正 Weka 运行时 `-SNAPSHOT` 与 Maven 版本不一致导致的安装后跳过问题。
- 修正工作目录 `data/` 被重复拼接成 `data/data/` 的路径错误。
- MCP 的 `class_index` 最小值为 1；界面 `0` 表示最后一列时不再发送该参数。
- 实验 ID 只在 EDT 回调中更新，避免后台线程修改 Swing 控件。
- 在通知成功回调前清除控制器忙碌状态，并修复按钮刷新竞态；界面回调异常会进入错误处理。
- GUI 测试保留有界面执行，子进程保持无界面运行，没有通过删除或跳过失败测试来绕过问题。

调试中还发现测试夹具将数据误传给分类面板、导出状态被误当成精确文本匹配等问题，均已修正；不能把这些测试错误归因于 Weka 或 MCP 服务。

## 已知边界

- 首版 GUI 覆盖数据加载/画像/拆分、分类器训练评估、预测、固定参数聚类、实验保存恢复及模型导出；其他 MCP 能力仍通过 STDIO 服务调用。
- 聚类器固定为 SimpleKMeans、3 簇；拆分比例和种子固定；尚未实现 AutoML、关联规则、可视化、完整搜索和原生结果列表的专属 GUI。
- 当前未验证 Windows/Linux 的真实 Explorer，也未验证 Weka 3.8.8 正式版或更高版本。跨平台路径与进程启动使用 Java API，但这不等于已完成这些平台的验收。
- `licenses/README.md` 只是审查起点。对外发布前仍需检查 fat JAR 内 Weka 和全部传递依赖的版权、许可证通知及源码提供义务。
- 底层服务日志出现 macOS ARM64 原生线性代数库的警告，但 Weka 使用回退实现，测试全部通过；该警告不等同于原生性能验证。
