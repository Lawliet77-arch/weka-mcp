# Third-party license material

每个以 Maven 坐标命名的目录对应该组件的实际版本：

- `declared.pom` 是该版本的原始 Maven POM，包含其直接声明的许可证、版权或项目链接。
- `inherited-parent.pom` 是许可证继承来源，仅在相关组件中保留。
- `META-INF-*` 等文件从对应二进制 JAR 原样提取，没有改写其中的版权。
- `source-*` 文件（如有）从同版本 Maven sources JAR 原样提取。
- 根目录的 `Apache-2.0.txt`、`LGPL-2.1.txt`、`LGPL-3.0.txt` 等是相关上游许可证文本；适用组件见 `THIRD_PARTY_NOTICES.md`。

本项目的 GPL-3.0 文本位于分发包的 `LICENSE`。本目录不表示所有组件采用相同许可证，也不替代第三方源码中的逐文件版权声明。
