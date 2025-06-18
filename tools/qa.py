from transformers import pipeline

def answer_question(context: str, question: str) -> str:
    qa = pipeline("question-answering", model="deepset/tinyroberta-squad2")
    return qa(question=question, context=context)["answer"]
