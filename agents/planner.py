from utils.chunking import chunk_text
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from utils.postprocess import preprocess_for_planner

class PlannerAgent:
    def __init__(self):
        # Switched to flan-t5-large for better planning
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=128)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        # Strengthened prompt with explicit resume summary example and instruction to ignore contact info
        self.prompt = PromptTemplate.from_template(
            """
You are an expert AI assistant. Given the following user task and document, break the task into clear, actionable steps. Each step should be a single, focused instruction. If the task has multiple parts, list each as a separate step.

When the document is a resume and the task is to summarize, focus on experience, education, and key skills. Ignore contact information, URLs, and boilerplate.

Example:
Document:
John Doe\njohn@email.com\n123-456-7890\nGitHub: github.com/johndoe\nLinkedIn: linkedin.com/in/johndoe\n\nExperienced software engineer with expertise in Python, machine learning, and cloud computing.\nEducation: B.Sc. in Computer Science, XYZ University\nSkills: Python, ML, AWS, Docker\n
Task: Summarize the document
Steps:
1. Write a concise summary of the candidate's experience, education, and key skills.

Document:
{doc}

Task: {task}
Steps:
"""
        )
        self.chain = RunnableSequence(self.prompt | llm)

    def run(self, task: str, doc: str) -> str:
        # Preprocess doc for planning
        doc = preprocess_for_planner(doc, max_chars=1000)
        prompt_input = {"task": task, "doc": doc}
        try:
            result = self.chain.invoke(prompt_input)
            return result
        except Exception as e:
            return f"[Error] Planning failed: {str(e)}"

    def structured_output(self, task: str, doc: str) -> dict:
        steps = self.run(task, doc)
        import re
        numbered = re.findall(r'\d+\.\s*([^\d]+?)(?=\d+\.|$)', steps, re.DOTALL)
        if numbered:
            step_lines = [s.strip().replace('\n', ' ') for s in numbered if s.strip()]
        else:
            step_lines = re.split(r'\b(?:also,|and)\b', steps, flags=re.IGNORECASE)
            step_lines = [s.strip().replace('\n', ' ') for s in step_lines if s.strip()]
        step_lines = [s for s in step_lines if len(s) > 2 and not all(c in '].,;:' for c in s)]
        # Fallback: if calculation pattern in task but not in steps, add it
        calc_patterns = [r'mean of \[.*?\]', r'median of \[.*?\]', r'stdev of \[.*?\]', r'calculate [^\.]+']
        for pat in calc_patterns:
            if re.search(pat, task, re.IGNORECASE):
                found = any(re.search(pat, s, re.IGNORECASE) for s in step_lines)
                if not found:
                    match = re.search(pat, task, re.IGNORECASE)
                    if match:
                        step_lines.append(match.group(0))
        def normalize_calc(s):
            s = re.sub(r'calculate|the|\s+', '', s, flags=re.IGNORECASE)
            s = s.lower()
            return s
        seen = set()
        unique_steps = []
        for s in step_lines:
            key = normalize_calc(s) if any(op in s.lower() for op in ['mean', 'median', 'stdev']) else s.lower()
            if key not in seen:
                unique_steps.append(s)
                seen.add(key)
        return {"task": task, "steps": unique_steps}
