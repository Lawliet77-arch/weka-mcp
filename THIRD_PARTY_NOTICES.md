# 第三方组件与源码通知

本项目源码采用 GPL-3.0-or-later，完整 GPL-3.0 文本见 `LICENSE`。Weka 的版权归 The University of Waikato 及相应贡献者所有；其他依赖保留各自的版权与许可证。本文件不把第三方组件重新声明为统一许可证。

以下清单以 `weka-mcp-java` 1.6.0 的实际运行时依赖解析结果为依据。GUI 0.8.0 另行携带的 Jackson 3.1.4 和 annotations 2.21 也列在其中。版本以 POM 和锁定构建为准；许可证名称来自实际 POM、继承 POM 或 JAR 内通知，不凭名称猜测条款版本。

## 运行时组件

| 组件（groupId:artifactId） | 版本 | 声明的许可证与来源 |
|---|---:|---|
| `nz.ac.waikato.cms.weka:weka-stable` | 3.8.7 | GPL-3.0-or-later；POM 为 GNU GPL 3，源码头明确“version 3 or later”。Weka 启动时可能显示 3.8.8-SNAPSHOT，不是另行升级的依赖版本。 |
| `io.modelcontextprotocol.sdk:mcp`, `mcp-core`, `mcp-json-jackson3` | 2.1.0 | MIT；[项目](https://github.com/modelcontextprotocol/java-sdk)，实际 v2.1.0 LICENSE 已保留。 |
| `tools.jackson.core:jackson-core`, `jackson-databind`; `tools.jackson.dataformat:jackson-dataformat-yaml` | 3.1.4 | Apache-2.0；保留各 JAR 的 LICENSE/NOTICE。 |
| `com.fasterxml.jackson.core:jackson-annotations` | 2.21 | Apache-2.0；保留 JAR 内 Jackson NOTICE。 |
| `com.networknt:json-schema-validator` | 3.0.6 | Apache License Version 2.0；[项目](https://github.com/networknt/json-schema-validator)。 |
| `com.ethlo.time:itu` | 1.14.0 | Apache-2.0；[项目](https://github.com/ethlo/itu)。 |
| `io.projectreactor:reactor-core` | 3.7.0 | Apache-2.0；[项目](https://github.com/reactor/reactor-core)。 |
| `org.reactivestreams:reactive-streams` | 1.0.4 | MIT-0；实际 v1.0.4 LICENSE 已保留。 |
| `org.snakeyaml:snakeyaml-engine` | 3.0.1 | Apache-2.0；[项目](https://github.com/snakeyaml/snakeyaml-engine)。 |
| `ch.qos.logback:logback-classic`, `logback-core` | 1.5.38 | EPL-2.0 **或** LGPL-2.1-only，见 logback-parent POM 的双许可证声明；两份文本均随包提供。 |
| `org.slf4j:slf4j-api` | 2.0.17 | MIT；JAR 内完整许可证已保留。 |
| `com.github.vbmacher:java-cup`, `java-cup-runtime` | 11b-20160615 | CUP Parser Generator Copyright Notice, License, and Disclaimer；BSD 风格自定义通知，不是 GNU GPL。[上游说明](http://www2.cs.tum.edu/projects/cup/install.php)。 |
| `nz.ac.waikato.cms.weka.thirdparty:bounce` | 0.18 | POM 链接 BSD-3-Clause，另有 BrowserLauncher 的 AS-IS 版权/再分发通知；两者均保留。 |
| `com.googlecode.matrix-toolkits-java:mtj` | 1.0.4 | LGPL-2.1-or-later；同版本源码头明确“version 2.1 or later”，示例原文及 LGPL-2.1 全文已保留。[项目](https://github.com/fommil/matrix-toolkits-java/)。 |
| `com.github.fommil:jniloader` | 1.1 | LGPL-3.0；上游 LICENSE 明确 Version 3，原 POM 仅写 LGPL。上游许可证全文已保留。[项目](https://github.com/fommil/jniloader)。 |
| `com.github.fommil.netlib:core`, `native_ref-java`, `native_system-java` | 1.1 | 继承 parent POM 的 BSD 3 Clause。 |
| `com.github.fommil.netlib:netlib-native_ref-{osx-x86_64,linux-x86_64,linux-i686,win-x86_64,win-i686,linux-armhf}` | 1.1 | 上述 native_ref 父项目许可证；保留对应 POM 与继承 POM。 |
| `com.github.fommil.netlib:netlib-native_system-{osx-x86_64,linux-x86_64,linux-i686,linux-armhf,win-x86_64,win-i686}` | 1.1 | 上述 native_system 父项目许可证；保留对应 POM 与继承 POM。 |
| `com.googlecode.netlib-java:netlib-java` | 1.1 | 继承 netlib parent POM 的 BSD 3 Clause。 |
| `net.sourceforge.f2j:arpack_combined_all` | 0.1 | POM 名称为 The BSD License，未明确 2/3 条款；以同版本源码为准。[项目](https://sourceforge.net/projects/f2j/)。 |
| `com.github.fracpete:jfilechooser-bookmarks` | 0.1.11 | GNU General Public License 3，按 POM 声明。 |
| `com.github.fracpete:jclipboardhelper` | 0.1.2 | GNU General Public License 3，按 POM 声明。 |
| `com.sun.istack:istack-commons-runtime` | 3.0.12 | JAR 内 Eclipse Distribution License 1.0（BSD 风格）及 NOTICE。 |
| `jakarta.activation:jakarta.activation-api`; `com.sun.activation:jakarta.activation` | 1.2.2 | JAR 内 Eclipse Distribution License 1.0 及 NOTICE。 |
| `jakarta.xml.bind:jakarta.xml.bind-api` | 2.3.3 | JAR 内 Eclipse Distribution License 1.0 及 NOTICE。 |
| `org.glassfish.jaxb:jaxb-runtime`, `txw2` | 2.3.9 | JAR 内 Eclipse Distribution License 1.0 及 NOTICE。 |
| `com.formdev:flatlaf` | 3.7.1 | Apache-2.0；JAR 内许可证已保留。 |

标准许可证全文及从各 JAR 提取的原始文件见 `third-party-notices/`。`declared.pom` 和必要时的 `inherited-parent.pom` 用于保留实际声明来源。该目录是上述版本的通知快照，不是未来依赖升级可复用的固定清单。

## 对应源码

发布附件 `weka-mcp-third-party-sources.zip` 按 Maven 坐标保存可获得的、同版本 `*-sources.jar` 和原始 POM，并附 SHA-256 清单。它们是上游发布的对应源码快照，未将本项目自己的源码混入其中。

也可从 Maven Central 逐项下载：

```text
https://repo.maven.apache.org/maven2/<groupId 的路径>/<artifactId>/<version>/<artifactId>-<version>-sources.jar
```

例如 Weka stable 3.8.7 的源码为：

https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.7/weka-stable-3.8.7-sources.jar

POM 使用 `-sources.jar` 相同的地址后缀。未发布 sources 附件的组件会在源码包清单中标记为未提供；其上游项目和 POM 链接仍保留，不能把不同版本的源码当作对应源码。构建期插件、测试库和本机 Maven 缓存路径不属于运行时二进制清单。

本文件是按声明版本进行的依赖通知整理，不是完整法律审计或对所有使用场景的法律意见。
