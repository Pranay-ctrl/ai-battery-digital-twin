# 🔔 AI Battery Digital Health & RUL Prognostics

A temporal deep learning digital twin predicting battery State-of-Health (SOH) and Remaining Useful Life (RUL) using the NASA Li-ion benchmark.

## Features
- Stacked LSTM architecture (64/32 units)
- Cross-cell Transfer Learning adapted for B0007
- Epistemic uncertainty bounds (±1.96ρ)
- Offline Streamlit dashboard running at port 8505


## Quickstart
```bash
pip3 install -r requirements.txt
python3 -m streamlit run app.py --server.port 8505
```
