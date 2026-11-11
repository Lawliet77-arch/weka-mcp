#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
  JAVA="$JAVA_HOME/bin/java"
else
  JAVA="$(command -v java)"
fi
if [[ -f "$BASE_DIR/weka-mcp-java-1.6.0.jar" ]]; then
  JAR="$BASE_DIR/weka-mcp-java-1.6.0.jar"
else
  JAR="$BASE_DIR/target/weka-mcp-java-1.6.0.jar"
fi

exec "$JAVA" \
  "-Xmx${WEKA_MCP_HEAP:-2g}" \
  -Dfile.encoding=UTF-8 \
  -Djava.awt.headless=true \
  "-Dweka.mcp.config=$BASE_DIR/config.properties" \
  -jar "$JAR"
