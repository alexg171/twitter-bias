@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM run_all.bat  —  Full regeneration of all analysis and figures
REM
REM Usage: double-click run_all.bat  OR  run from command prompt
REM ─────────────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo ============================================================
echo   FULL ANALYSIS REGENERATION
echo ============================================================

echo.
echo [ 1/5 ]  Classifying topics --^> unique_topics.csv
echo ------------------------------------------------------------
python twitter_unique.py
if errorlevel 1 echo WARNING: Step 1 had errors, continuing...

echo.
echo [ 2/5 ]  Category composition charts
echo ------------------------------------------------------------
python category_analysis.py
if errorlevel 1 echo WARNING: Step 2 had errors (CSV open in Excel?), continuing...

echo.
echo [ 3/5 ]  Category DiD (main results)
echo ------------------------------------------------------------
python category_did.py
if errorlevel 1 echo WARNING: Step 3 had errors, continuing...

echo.
echo [ 4/5 ]  Parallel trends + event study plots
echo            (takes 2-3 minutes for bootstrapping)
echo ------------------------------------------------------------
python category_plots.py
if errorlevel 1 echo WARNING: Step 4 had errors, continuing...

echo.
echo [ 5/5 ]  Demographic visualizations
echo ------------------------------------------------------------
python category_demographics.py
if errorlevel 1 echo WARNING: Step 5 had errors, continuing...

echo.
echo ============================================================
echo   ALL DONE
echo.
echo   Key outputs:
echo     out\unique_topics.csv
echo     out\category_did_results.csv
echo     out\category_counts.csv
echo     out\figures\category_shift.png
echo     out\figures\category_did.png
echo     out\figures\parallel_trends\  (17 plots)
echo     out\figures\event_study\      (17 plots)
echo     out\figures\demographics\     (summary + scatter)
echo ============================================================
echo.
pause
