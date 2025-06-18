# AGI Agent Playground

A modular, multi-agent pipeline for document understanding and reasoning using only free, local Hugging Face models (no API keys required).

## Features
- PDF document reading and extraction
- Task planning and decomposition
- Summarization, keyword extraction, calculation, and question answering
- Critique and improvement of outputs
- Clean Streamlit UI for user interaction
- Download results as Markdown or PDF
- User feedback collection

## Quick Start
1. Clone this repo and install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```
3. Upload a PDF and enter any task or question (e.g., "Summarize the document", "What is the candidate's GPA?").

## Example Tasks
- Summarize this resume and highlight key skills.
- What programming languages are listed?
- Calculate the mean of [85, 90, 78, 92].

## Project Structure
- `app.py` - Main pipeline logic
- `streamlit_app.py` - Streamlit UI
- `agents/` - Agent classes (Reader, Planner, Tools, Critic, Improver)
- `tools/` - Tool implementations (summarizer, keywords, calculations, QA)
- `utils/` - Utilities (chunking, postprocessing, export)

## License
MIT
