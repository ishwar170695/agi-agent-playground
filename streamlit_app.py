import streamlit as st
import os
import json
from agents.ReaderAgent import ReaderAgent
from agents.planner import PlannerAgent
from agents.ToolsAgent import ToolAgent
from agents.CriticAgent import CriticAgent
from agents.ImproverAgent import ImproverAgent
from utils.postprocess import clean_and_format_summary, preprocess_for_planner
from utils.export import results_to_markdown
import io
from fpdf import FPDF

def run_pipeline(task, doc):
    # Preprocess doc for planning
    doc_for_planner = preprocess_for_planner(doc, max_chars=1000)
    planner = PlannerAgent()
    tool = ToolAgent()
    critic = CriticAgent()
    improver = ImproverAgent()
    # Pass both task and doc to planner
    structured_steps = planner.structured_output(task, doc_for_planner)
    final_output = []
    for step in structured_steps["steps"]:
        # Add more flexible matching for tool routing
        step_lower = step.lower()
        if any(x in step_lower for x in ["calculate", "mean", "median", "stdev"]):
            import re
            expr_match = re.search(r"\[(.*?)\]", step)
            if expr_match:
                numbers = expr_match.group(1)
                expr = f"mean([{numbers}])" if "mean" in step_lower else numbers
                output = tool.run(step, expr)
            else:
                output = tool.run(step, step)
        elif any(x in step_lower for x in ["summarize", "summary", "write a resume", "write summary", "short summary", "concise summary"]):
            output = tool.run("summarize", doc)
        elif any(x in step_lower for x in ["keywords", "key skills", "key points", "extract skills", "list skills"]):
            output = tool.run("extract_keywords", doc)
        elif any(x in step_lower for x in ["question", "answer", "qa"]):
            # For QA, try to extract the question from the step
            import re
            q_match = re.search(r'question[:\-]?\s*(.*)', step, re.IGNORECASE)
            question = q_match.group(1) if q_match else step
            output = tool.run("answer_question", doc, question=question)
        else:
            output = tool.run(step, doc)
        output = clean_and_format_summary(output, mode="general")
        critique = critic.critique(output, context=doc)
        critique = clean_and_format_summary(critique, mode="general")
        improved = improver.improve(output, critique, context=doc)
        improved = clean_and_format_summary(improved, mode="general")
        final_output.append({"step": step, "output": output, "critique": critique, "improved": improved})
    return final_output

def save_feedback(i, item, rating, comments):
    feedback = {
        "step": item['step'],
        "output": item['output'],
        "critique": item['critique'],
        "improved": item['improved'],
        "rating": rating,
        "comments": comments
    }
    feedback_path = os.path.join("feedback", f"feedback_step_{i}.json")
    os.makedirs("feedback", exist_ok=True)
    with open(feedback_path, "w", encoding="utf-8") as fb:
        json.dump(feedback, fb, indent=2)

def normalize_text(text):
    return (
        text.replace('\u2013', '-')
            .replace('\u2014', '-')
            .replace('\u2019', "'")
            .replace('\u201c', '"')
            .replace('\u201d', '"')
            .encode('latin-1', errors='replace').decode('latin-1')
    )

st.title("AGI Agent Pipeline Demo")
st.markdown("""
Upload a PDF document and enter any task or question. The pipeline will:
- Read and analyze your document
- Plan and decompose your task
- Use local tools (summarizer, QA, calculator, etc.)
- Critique and improve each output
- Let you download results and give feedback
""")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
    st.divider()
    task = st.text_input(
        "Task",
        placeholder="Type your task or question here (e.g., 'Summarize the document', 'What is the candidate's GPA?')"
    )
    st.info("Enter any instruction or question. Example: 'Summarize the document', 'What programming languages are listed?', 'Calculate the mean of [85, 90, 78, 92]'.")
    if st.button("Run AGI Pipeline"):
        with st.spinner("Running pipeline. This may take a moment..."):
            try:
                temp_path = os.path.join("examples", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                reader = ReaderAgent()
                doc = reader.run(temp_path)
                final_output = run_pipeline(task, doc)
                st.header("Results")
                for i, item in enumerate(final_output, 1):
                    st.subheader(f"Step {i}: {item['step']}")
                    with st.expander("Main Output"):
                        st.markdown(item['output'])
                    with st.expander("Critique"):
                        st.markdown(item['critique'])
                    st.markdown("**Improved Output:**")
                    st.markdown(item['improved'])
                    st.write("### Your Feedback")
                    rating = st.radio(f"How useful was the improved output for Step {i}?", ["Very useful", "Somewhat useful", "Not useful"], key=f"rating_{i}")
                    comments = st.text_area(f"Comments for Step {i}", key=f"comments_{i}")
                    if st.button(f"Submit Feedback for Step {i}"):
                        save_feedback(i, item, rating, comments)
                        st.success("Feedback submitted!")
                if final_output:
                    md_content = results_to_markdown(final_output)
                    st.download_button(
                        label="Download Results as Markdown",
                        data=md_content,
                        file_name="agi_pipeline_results.md",
                        mime="text/markdown"
                    )
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    for line in md_content.split('\n'):
                        pdf.multi_cell(0, 10, normalize_text(line))
                    pdf_bytes = pdf.output(dest='S').encode('latin1')
                    st.download_button(
                        label="Download Results as PDF",
                        data=pdf_bytes,
                        file_name="agi_pipeline_results.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.text(traceback.format_exc())
else:
    st.info("Please upload a PDF document to get started.")
