@echo off
setlocal
set "PACKAGE_HOME=%~dp0.."
for %%I in ("%PACKAGE_HOME%") do set "PACKAGE_HOME=%%~fI"
java -Dfile.encoding=UTF-8 -cp "%PACKAGE_HOME%\weka-mcp-gui-@project.version@.jar;%PACKAGE_HOME%\lib\*" weka.mcp.gui.McpBridgeConnector %*
exit /b %ERRORLEVEL%
