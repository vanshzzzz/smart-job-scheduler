@echo off
cd /d "%~dp0"
set PY=python
where python >NUL 2>NUL || set PY=py
echo Installing requirements (first run only takes a minute)...
%PY% -m pip install -q -r requirements.txt
echo.
echo Starting Smart Job Scheduling System at http://localhost:8501
echo Close this window to stop the app.
start "" cmd /c "timeout /t 8 >NUL & start http://localhost:8501"
%PY% -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
