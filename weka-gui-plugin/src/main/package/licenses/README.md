# 许可证与第三方通知

本包包含多个许可证的组件，不能把它们都视为采用 Weka 的 GPL 许可证。

- `../LICENSE`：本项目的 GNU GPL version 3 文本。
- `../THIRD_PARTY_NOTICES.md`：按 Maven 声明版本整理的运行时依赖、许可证、版权及对应源码获取说明。
- `third-party-notices/`：从实际依赖 JAR 提取的许可证与 NOTICE，以及按组件补充的上游许可证文本。保留各组件自己的版权归属。
- `../tools/weka-mcp-java-1.6.0.jar`：独立 STDIO 后台服务，内含 Weka、MCP Java SDK、Jackson、Logback 及传递依赖；其 JAR 内也携带相同的许可证资料。

Weka 的版权归 The University of Waikato 及相应贡献者所有。后台服务由独立 JVM 启动，这种进程划分不改变任何第三方组件的许可证或源码提供义务。

发布时提供 `weka-mcp-third-party-sources.zip`，包含可获得的、与声明版本对应的 Maven `sources` JAR 及原始 POM；未取得源码附件的项目在通知清单中明确标出。本资料是依赖通知整理，不是完整法律审计。
