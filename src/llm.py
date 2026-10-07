"""
LLM Generation Module supporting local HuggingFace Seq2Seq model (flan-t5-small)
and optional API providers (Gemini / OpenAI) if keys are provided.
"""

import os
from typing import Optional


class LocalFlanT5Generator:
    """Lightweight local generator running on CPU via HuggingFace transformers."""
    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class LLMClient:
    def __init__(self, provider: str = "local", model_name: Optional[str] = None):
        """
        Initializes the LLM interface. Defaults to local flan-t5-small if no API key is present.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if provider == "gemini" and gemini_key:
            self.provider = "gemini"
            from google import genai
            self.client = genai.Client(api_key=gemini_key)
            self.model_name = model_name or "gemini-2.5-flash"
        elif provider == "openai" and openai_key:
            self.provider = "openai"
            from openai import OpenAI
            self.client = OpenAI(api_key=openai_key)
            self.model_name = model_name or "gpt-4o-mini"
        else:
            self.provider = "local"
            self.generator = LocalFlanT5Generator(model_name or "google/flan-t5-small")

    def generate_base_answer(self, query: str) -> str:
        """
        Asks the query to the base LLM without any retrieved context.
        Demonstrates the base LLM's inability to answer proprietary or novel questions.
        """
        prompt = (
            f"Answer the following question accurately:\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return self._generate(prompt)

    def generate_rag_answer(self, query: str, context: str) -> str:
        """
        Asks the query to the LLM grounded in retrieved context.
        """
        prompt = (
            f"Answer the question based strictly on the provided context below. "
            f"If the context does not contain the answer, state that the information is unavailable.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        else:
            return self.generator.generate(prompt)

