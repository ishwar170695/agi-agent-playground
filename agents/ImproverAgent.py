from transformers import pipeline, AutoTokenizer

class ImproverAgent:
    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.improver = pipeline("text2text-generation", model=model_id, tokenizer=self.tokenizer, max_new_tokens=256)

    def improve(self, output: str, critique: str, context: str = "") -> str:
        prompt = (
            "You are an expert editor. Given the following output and critique, rewrite the output to address all issues and suggestions. "
            "Make the improved output clear, accurate, and well-structured.\n\n"
            "Example Critique: The summary is too verbose and lacks focus on key skills.\n"
            "Example Improved Output: A concise summary focused on the candidate's main skills and achievements.\n\n"
        )
        # Use only a short context if available
        if context and len(context) < 500:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Original Output:\n{output}\n\nCritique:\n{critique}\n\nImproved Output:"
        try:
            result = self.improver(prompt)[0]["generated_text"]
            return result.strip()
        except Exception as e:
            return f"[Error] Improvement failed: {str(e)}"
