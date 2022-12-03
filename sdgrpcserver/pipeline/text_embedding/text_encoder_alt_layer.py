from typing import Literal


class TextEncoderAltLayer:
    def __init__(
        self,
        text_encoder,
        layer: Literal["final", "penultimate"] | int = "final",
    ):
        self.text_encoder = text_encoder
        self.layer = layer

    def __call__(self, input_ids):
        text_embeddings = self.text_encoder(
            input_ids,
            output_hidden_states=(self.layer != "final"),
            return_dict=True,
        )

        if self.layer == "final":
            res = text_embeddings.last_hidden_state
        elif self.layer == "penultimate":
            res = self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[-2]
            )
        else:
            res = self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[self.layer]
            )

        # text_encoder clients expect tuple of (final layer, pool)
        return (res, None)
