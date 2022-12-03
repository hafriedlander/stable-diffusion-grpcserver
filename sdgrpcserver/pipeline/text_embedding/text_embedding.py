class TextEmbedding:
    def __init__(self, pipe, text_encoder, **kwargs):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = text_encoder
        self.device = pipe.execution_device

    def get_text_embeddings(self, prompt):
        raise NotImplementedError("Not implemented")

    def get_uncond_embeddings(self, prompt):
        raise NotImplementedError("Not implemented")

    def get_embeddings(self, prompt, uncond_prompt=None):
        """Prompt and negative a both expected to be lists of strings, and matching in length"""
        text_embeddings = self.get_text_embeddings(prompt)
        uncond_embeddings = (
            self.get_uncond_embeddings(uncond_prompt)
            if uncond_prompt is not None
            else None
        )

        return (text_embeddings, uncond_embeddings)

    def repeat(self, embedding, count):
        bs_embed, seq_len, _ = embedding.shape
        embedding = embedding.repeat(1, count, 1)
        embedding = embedding.view(bs_embed * count, seq_len, -1)

        return embedding
