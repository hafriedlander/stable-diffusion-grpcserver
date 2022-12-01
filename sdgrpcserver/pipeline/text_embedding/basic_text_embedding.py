from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .text_embedding import TextEmbedding


class BasicTextEmbedding(TextEmbedding):
    def __init__(self, pipe, text_encoder, layer="final", **kwargs):
        super().__init__(pipe, text_encoder, **kwargs)
        self.layer = layer

    def _get_embeddedings(self, strings, label):
        tokenizer = self.pipe.tokenizer

        max_length = min(
            tokenizer.model_max_length,
            self.text_encoder.config.max_position_embeddings,
        )

        # get prompt text embeddings
        text_inputs = tokenizer(
            strings,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > max_length:
            removed_text = tokenizer.batch_decode(text_input_ids[:, max_length:])
            logger.warning(
                f"The following part of your {label} input was truncated because CLIP can only handle sequences up to "
                f"{max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, :max_length]

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.pipe.device),
            output_hidden_states=(self.layer != "final"),
            return_dict=True,
        )

        if self.layer == "final":
            return text_embeddings.last_hidden_state
        elif self.layer == "penultimate":
            return self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[-2]
            )
        else:
            return self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[self.layer]
            )

    def get_text_embeddings(self, prompt):
        return self._get_embeddedings(prompt.as_unweighted_string(), "prompt")

    def get_uncond_embeddings(self, prompt):
        return self._get_embeddedings(prompt.as_unweighted_string(), "negative prompt")
