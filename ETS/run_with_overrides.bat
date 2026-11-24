@echo off
setlocal
c:/venv/OpAn311/Scripts/python update_config.py config.json "{\"seed\":666,\"numCustomers\":100,\"h\":[0,1,0,0],\"isUseCase\":true}"
if errorlevel 1 exit /b 1
c:/venv/OpAn311/Scripts/python main.py