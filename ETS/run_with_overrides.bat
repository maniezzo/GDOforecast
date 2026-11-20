@echo off
setlocal

REM --- 1. Define the path to your JSON and your main program ---
set "CONFIG_FILE=config.json"
set "YOUR_PROGRAM=path\to\your_program.exe"  REM <-- CHANGE THIS

REM --- 2. Define the parameters you want to override ---
REM    Use PowerShell syntax. Strings need quotes (e.g., 'new_value').
set "OVERRIDES={ seed: 123, numCustomers: 60, h: [0,20,0,0], isUseCase: $true }"

REM --- 3. Use PowerShell to read, modify, and write the JSON ---
echo Updating configuration...
powershell -NoProfile -Command ^
  "$json = Get-Content '%CONFIG_FILE%' -Raw | ConvertFrom-Json;" ^
  "$overrides = %OVERRIDES%;" ^
  "foreach ($prop in $overrides.PSObject.Properties) {" ^
  "    $json.($prop.Name) = $prop.Value;" ^
  "}" ^
  "$json | ConvertTo-Json -Depth 10 | Set-Content -Path '%CONFIG_FILE%' -Encoding UTF8"

if %errorlevel% neq 0 (
    echo Failed to update JSON configuration. Exiting.
    exit /b %errorlevel%
)
echo Configuration updated successfully.

REM --- 4. Call your program ---
echo Running your program...
call "%YOUR_PROGRAM%"