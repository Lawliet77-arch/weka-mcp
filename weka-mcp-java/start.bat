@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 2
set "JAR=weka-mcp-java-1.6.0.jar"
if not exist "%JAR%" set "JAR=target\weka-mcp-java-1.6.0.jar"
if not exist "%JAR%" (
  echo [weka-mcp-java] Fat jar not found: "%JAR%" 1>&2
  popd
  exit /b 2
)
set "JAVA="
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" set "JAVA=%JAVA_HOME%\bin\java.exe"
if not defined JAVA set "JAVA=java.exe"
if not defined WEKA_MCP_HEAP set "WEKA_MCP_HEAP=2g"
"%JAVA%" "-Xmx%WEKA_MCP_HEAP%" -Dfile.encoding=UTF-8 -Djava.awt.headless=true "-Dweka.mcp.config=config.properties" -jar "%JAR%"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
