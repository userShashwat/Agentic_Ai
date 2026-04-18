# HR Policy Bot – Agentic AI Capstone Project

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Get a free Gemini API key from https://aistudio.google.com/app/apikey
3. Paste the key in `hr_agent.py` (replace `YOUR_GEMINI_API_KEY`)
4. Run the agent: `python hr_agent.py` (to test)
5. Launch web UI: `streamlit run hr_streamlit.py`

## Features
- RAG with 10 HR policy documents
- Memory across conversation turns
- Self-evaluation (faithfulness score, auto-retry)
- Tool support (date/time)
- Streamlit chat interface