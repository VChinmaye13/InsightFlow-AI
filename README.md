# InsightFlow AI — Intelligent Data Analyst Copilot

A CPU-friendly Streamlit application for interactive dataset exploration.

## Features
- CSV / Excel upload and preview
- Dataset profile and data-quality audit
- Interactive Plotly charts
- Local question helper for common statistics
- Optional OpenAI API-assisted explanation (API key entered per session)
- Isolation Forest anomaly detection
- Downloadable Markdown report and CSV results

## Quick start (Windows / Anaconda Prompt)
```bash
cd path\to\InsightFlow_AI
conda create -n insightflow python=3.11 -y
conda activate insightflow
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
The app opens at http://localhost:8501.

## Demo
The app includes a synthetic sample dataset when no file is uploaded. Upload your own CSV or Excel file using the sidebar.

## Optional AI mode
The optional AI question box accepts an API key for the current session. Do not save API keys in source code or commit them to GitHub. AI responses are based on the schema and a small sample, so use the deterministic local tools for exact computed statistics.

## Deploy
Push this folder to GitHub, then create a Streamlit Community Cloud app using `app.py` as the entrypoint. Add secrets only in the hosting platform's secrets settings. See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Limitations
- Anomaly detection is unsupervised and flags unusual numeric rows; it is not proof of fraud or error.
- Natural-language local mode supports common, transparent analytical question patterns. Open-ended AI mode is optional.
- Large files may require sampling or more memory.
