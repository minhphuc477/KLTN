@echo off
setlocal
cd /d "%~dp0"
echo [1/5] XeLaTeX pass 1...
xelatex -interaction=nonstopmode IEEE_FULL_PAPER_VN.tex >nul
if errorlevel 1 goto :failed
echo [2/5] BibTeX...
bibtex IEEE_FULL_PAPER_VN >nul
if errorlevel 1 goto :failed
echo [3/5] XeLaTeX pass 2...
xelatex -interaction=nonstopmode IEEE_FULL_PAPER_VN.tex >nul
if errorlevel 1 goto :failed
echo [4/5] XeLaTeX pass 3...
xelatex -interaction=nonstopmode IEEE_FULL_PAPER_VN.tex >nul
if errorlevel 1 goto :failed
echo [5/5] XeLaTeX pass 4 (final)...
xelatex -interaction=nonstopmode IEEE_FULL_PAPER_VN.tex
if errorlevel 1 goto :failed
echo.
echo Build finished: IEEE_FULL_PAPER_VN.pdf
exit /b 0
:failed
echo Build failed. See IEEE_FULL_PAPER_VN.log
exit /b 1
