from transformers import pipeline, AutoTokenizer

class CriticAgent:
    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.critic = pipeline("text2text-generation", model=model_id, tokenizer=self.tokenizer, max_new_tokens=128)

    def critique(self, output: str, context: str = "") -> str:
        prompt = (
            "You are an expert reviewer. Critique the following output for factual accuracy, structure, and clarity. "
            "Be specific and actionable. If there are issues, suggest concrete improvements.\n\n"
            "Example Output: The summary is too verbose and lacks focus on key skills.\n"
            "Example Critique: The summary should be more concise and highlight the candidate's main skills and achievements.\n\n"
        )
        # Use only a short context if available
        if context and len(context) < 500:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Output:\n{output}\n\nCritique:"
        try:
            result = self.critic(prompt)[0]["generated_text"]
            return result.strip()
        except Exception as e:
            return f"[Error] Critique failed: {str(e)}"
