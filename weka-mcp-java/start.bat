@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "BASE_DIR=%~dp0"
set "JAVA="
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" set "JAVA=%JAVA_HOME%\bin\java.exe"
if not defined JAVA set "JAVA=java"
set "JAR=%BASE_DIR%weka-mcp-java-1.6.0.jar"
if not exist "%JAR%" set "JAR=%BASE_DIR%target\weka-mcp-java-1.6.0.jar"
if not exist "%JAR%" (
  echo [weka-mcp-java] Fat jar not found: "%JAR%" 1>&2
  exit /b 2
)
if not defined WEKA_MCP_HEAP set "WEKA_MCP_HEAP=2g"
"%JAVA%" "-Xmx%WEKA_MCP_HEAP%" -Dfile.encoding=UTF-8 -Djava.awt.headless=true "-Dweka.mcp.config=%BASE_DIR%config.properties" -jar "%JAR%"
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
