# Weka MCP GUI 0.7.0 原生可视化验证报告

## 结论

在既有原生应用自动化桥接基础上，新增真实 Weka **原生散点图、PNG 图像内容和独立图窗关闭**。Agent 可以为当前 Explorer 数据设置 X/Y 和颜色属性，读取图片并继续控制分析任务。

最终 `mvn -B -ntp clean verify` 为 **BUILD SUCCESS**：**16 个单元测试、3 个集成测试，合计 19 个测试全部通过，0 失败、0 错误、0 跳过**。

## 交付物

- 安装包：`weka-gui-plugin/target/weka-mcp-gui-0.7.0.zip`
- SHA-256：`07670d7a9dac41415ce566b9de85073db3281f7a6fd74f1f9db973a289d032f9`
- 原生散点图：`weka-gui-plugin/target/native-visualization.png`
- 安装与 Agent 配置说明：安装包根目录 `README.md`
- 前期验证：`NATIVE_BRIDGE_VALIDATION.md`、`EXPLORER_DATA_VALIDATION.md`、`CLUSTER_VALIDATION.md`、`ASSOCIATION_VALIDATION.md`、`ATTRIBUTE_SELECTION_VALIDATION.md`

## 新增工具

工具发现现返回 **43 个后台工具 + 21 个原生工具，共 64 个**。新增：

- `weka_explorer_visualize`：打开 Weka 原生散点图，可指定 X、Y 和颜色属性。
- `weka_explorer_capture_visualization`：返回图像元数据、文件路径和 `type=image`、`mimeType=image/png` 的 Base64 内容。
- `weka_explorer_close_visualization`：关闭桥接自己创建的图窗，不影响其他 Weka 窗口。

默认 X/Y 为前两个属性，颜色默认类别列（无类别列时为最后属性），空颜色属性可禁用颜色。宽度支持 320～1600，高度支持 240～1200，单张 PNG 限制 2 MB，默认保存到工作目录 `data/exports/visualizations/`。

`get_state` 的 `visualization_open` 表示桥接图窗状态，**不是后台训练任务**。图像是原生组件渲染，不是整个屏幕的截图。

## 实际验证

- 在真实 Explorer 中使用 Iris 数据，设置 X=`petallength`、Y=`petalwidth`、颜色=`class`，打开原生散点图。
- 获取 PNG，解析返回的 Base64 字节，确认可读取为有效图片，尺寸为 800×600。
- 比较图像内容与导出文件字节，结果一致。
- 图像中可见数据点、类别颜色、坐标轴及图例，不是空白占位图。
- 外部 Java MCP 客户端使用 X=`sepallength`、Y=`petalwidth`、颜色=`class`，验证 `type=image` 和 `mimeType=image/png`，并成功解析图像。
- 未先打开图窗就截图时，返回明确错误。
- 关闭图窗后 `visualization_open=false`，Explorer 和后台 MCP 仍可继续执行其他任务。
- 原有分类、聚类、关联、属性选择、画像、撤销、后台分析、模型保存恢复及安装流程继续通过回归。

## 边界

- 数据变化后需要重新调用可视化以刷新图像；没有自动的数据版本刷新机制。
- 当前只覆盖数据集散点图，不覆盖分类误差、聚类分配、模型内部结构或任意第三方对话框。
- 图像尺寸、字节数和路径均有边界，禁止通过无界导出拖垮 Agent 或宿主进程。
- 关闭只释放桥接创建的图窗，不会关闭用户已有的其他 Weka 窗口。
- 验证环境为 macOS ARM64、Temurin JDK 17.0.21+9、Maven 3.9.16、`weka-stable:3.8.7`（运行时自报 `3.8.8-SNAPSHOT`）。Windows `.bat` 已打包但未实测。
- 发布前仍需完成依赖许可证、版权通知及源码提供义务审查。
