from utils.chunking import chunk_text
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class PlannerAgent:
    def __init__(self):
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=128)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        # Improved prompt with few-shot example
        self.prompt = PromptTemplate.from_template(
            """
You are an expert AI assistant. Break the following task into clear, actionable steps. Each step should be a single, focused instruction. If the task has multiple parts, list each as a separate step.

Example:
Task: Summarize this resume and highlight key skills. Also, calculate the mean of [85, 90, 78, 92].
Steps:
1. Write a short summary of your experience and skills. Highlight the most important skills and accomplishments.
2. Calculate the mean of [85, 90, 78, 92].

Task: {input}
Steps:
"""
        )
        self.chain = RunnableSequence(self.prompt | llm)

    def run(self, task: str) -> str:
        max_input_tokens = 512
        chunks = chunk_text(task, max_chunk_tokens=max_input_tokens-20, overlap=0)
        results = []
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            print(f"[PlannerAgent Debug] Chunk token count: {token_count}")
            prompt_input = {"input": chunk}
            try:
                result = self.chain.invoke(prompt_input)
                results.append(result)
            except Exception as e:
                results.append(f"[Error] Planning failed: {str(e)}")
        if len(results) > 1:
            combined = " ".join(results)
            return self.run(combined)
        return results[0] if results else ""

    def structured_output(self, task: str) -> dict:
        steps = self.run(task)
        import re
        # Improved: split on numbered steps, and also on 'Also,' or 'and' for compound instructions
        numbered = re.findall(r'\d+\.\s*([^\d]+?)(?=\d+\.|$)', steps, re.DOTALL)
        if numbered:
            step_lines = [s.strip().replace('\n', ' ') for s in numbered if s.strip()]
        else:
            # Fallback: split on 'Also,' or ' and ' for compound instructions
            step_lines = re.split(r'\b(?:also,|and)\b', steps, flags=re.IGNORECASE)
            step_lines = [s.strip().replace('\n', ' ') for s in step_lines if s.strip()]
        # Remove any steps that are just punctuation or empty
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
        # Remove duplicate steps (case-insensitive, normalize calculation steps)
        def normalize_calc(s):
            # Remove 'calculate', 'the', and extra spaces, focus on numbers and operation
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
