from tools.summarizer import summarize_text
from tools.keywords import extract_keywords
from tools.calculations import calculate_expression
from tools.qa import answer_question
from utils.chunking import chunk_text
import nltk
nltk.download("punkt", quiet=True)

def summarize_with_chunking(text):
    return "\n\n".join(summarize_text(chunk).strip() for chunk in chunk_text(text))

def answer_with_context(context, question):
    return answer_question(context, question)

class ToolAgent:
    def __init__(self):
        self.available_tools = {
            "summarize": summarize_with_chunking,
            "extract_keywords": extract_keywords,
            "calculate": calculate_expression,
            "answer_question": answer_with_context,
        }

    def run(self, instruction: str, input_text: str, question: str = None) -> str:
        tool = self._route(instruction)
        if not tool:
            return f"[Error] No suitable tool found for instruction: {instruction}"
        if tool == answer_with_context:
            if question is None:
                return "[Error] No question provided for QA."
            return tool(input_text, question)
        return tool(input_text)

    def _route(self, instruction: str):
        instr = instruction.lower()
        if "summarize" in instr or "summary" in instr:
            return self.available_tools["summarize"]
        if "keywords" in instr or "key points" in instr:
            return self.available_tools["extract_keywords"]
        if any(x in instr for x in ["calculate", "math", "mean", "median", "stdev"]):
            return self.available_tools["calculate"]
        if any(x in instr for x in ["question", "answer", "qa"]):
            return self.available_tools["answer_question"]
        return None
