#!/bin/sh
set -eu
PACKAGE_HOME=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec java -Dfile.encoding=UTF-8 -cp "$PACKAGE_HOME/weka-mcp-gui-@project.version@.jar:$PACKAGE_HOME/lib/*" weka.mcp.gui.McpBridgeConnector "$@"
