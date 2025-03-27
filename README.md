Features:

- Upload CSV/XLSX files
- Preview top N rows
- Ask AI questions about your data
- Prompt history
- Feedback collection

Stack

- Frontend + Backend: Streamlit
- AI Layer: PandasAI + OpenAI GPT
- Data handling: Pandas

Run Locally

install dependencies:
pip install -r requirements.txt
streamlit run app.py

create a file called .env
type in:
OPENAI_API_KEY=sk... {your OpenAI API key}

#extra note
if the delete and the reuse button does not work, press it one more time.
As for reuse after the second press if it does not work navigate to the custom query section under operations manually
