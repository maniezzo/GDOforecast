@echo off
setlocal
c:/venv/OpAn311/Scripts/python update_config.py config.json "{\"seed\":123,\"numCustomers\":60,\"h\":[0,20,0,0],\"isUseCase\":false}"
if errorlevel 1 exit /b 1
c:/venv/OpAn311/Scripts/python main.py