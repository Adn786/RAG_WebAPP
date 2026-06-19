"""Wraps the GitHub Models (Azure inference) endpoint for embeddings and chat generation."""
import os

import tiktoken
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY from the provided context. "
    "Cite sources as [1], [2]. If the context is insufficient, say \"I don't know\"."
)


class LLMService:
    def __init__(self):
        github_pat = os.getenv("GITHUB_PAT")
        if not github_pat:
            raise ValueError("GITHUB_PAT environment variable not set")
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_pat,
        )
        self.generation_model = "gpt-4o"
        self.embedding_model = "text-embedding-3-small"
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_embeddings(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.embedding_model)
        return [item.embedding for item in response.data]

    def generate(self, messages, stream=True):
        return self.client.chat.completions.create(
            model=self.generation_model,
            messages=messages,
            stream=stream,
            temperature=0.0,
        )

    def build_messages(self, question: str, context_chunks: list):
        """context_chunks: list of dicts with 'ref' (1-based citation index), 'text',
        'source', and optional 'page'."""
        if not context_chunks:
            context_block = "(no relevant context found)"
        else:
            lines = []
            for c in context_chunks:
                page_part = f", page {c['page']}" if c.get("page") else ""
                lines.append(f"[{c['ref']}] (source: {c['source']}{page_part})\n{c['text']}")
            context_block = "\n\n".join(lines)

        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
