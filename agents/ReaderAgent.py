import fitz  # PyMuPDF

class ReaderAgent:
    def run(self, filepath: str) -> str:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text.strip()
