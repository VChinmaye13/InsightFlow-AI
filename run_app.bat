@echo off
call conda activate insightflow
python -m streamlit run app.py
pause
