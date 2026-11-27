@echo off
setlocal
set "JSON={""seed"":666,""numRep"":3,""numCustomers"":100,"
set "JSON=%JSON%""numCenters"":5,"
set "JSON=%JSON%""totdisp"":1.05,"
set "JSON=%JSON%""h"":[0,0,0,0],"
set "JSON=%JSON%""isUseCase"":false}"
c:/venv/OpAn311/Scripts/python update_config.py config.json "%JSON%"
if errorlevel 1 exit /b 1
c:/venv/OpAn311/Scripts/python main.py
set "JSON={""seed"":666,""numRep"":3,""numCustomers"":100,"
set "JSON=%JSON%""numCenters"":5,"
set "JSON=%JSON%""totdisp"":1.10,"
set "JSON=%JSON%""h"":[0,0,0,0],"
set "JSON=%JSON%""isUseCase"":false}"
c:/venv/OpAn311/Scripts/python update_config.py config.json "%JSON%"
if errorlevel 1 exit /b 1
c:/venv/OpAn311/Scripts/python main.py
set "JSON={""seed"":666,""numRep"":3,""numCustomers"":100,"
set "JSON=%JSON%""numCenters"":5,"
set "JSON=%JSON%""totdisp"":1.20,"
set "JSON=%JSON%""h"":[0,0,0,0],"
set "JSON=%JSON%""isUseCase"":false}"
c:/venv/OpAn311/Scripts/python update_config.py config.json "%JSON%"
if errorlevel 1 exit /b 1
c:/venv/OpAn311/Scripts/python main.py