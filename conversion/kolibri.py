from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1-BF16")
class KolibriModel(TextModel):
    model_arch = gguf.MODEL_ARCH.KOLIBRI

    def set_vocab(self) -> None:
        # byte-level BPE; the pre-tokenizer hash resolves to "kolibri"
        self._set_vocab_gpt2()

    def set_gguf_parameters(self) -> None:
        # the base class writes block count, embedding length, head counts, key/value length, rms eps, expert counts and rope base
        super().set_gguf_parameters()
        hparams = self.hparams

        # the sliding-window layers rotate the full head
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        # one ungated shared expert per layer
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["shared_expert_intermediate_size"])

        # 513 = 512 previous tokens + the current token (LLAMA_SWA_TYPE_STANDARD)
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        is_swa = [t == "sliding_attention" for t in hparams["layer_types"]]
        self.gguf_writer.add_sliding_window_pattern(is_swa)
        # the full-attention layers use no RoPE
        self.gguf_writer.add_rope_pattern(is_swa)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the generic TensorNameMap maps the sandwich norms to wrong tensors and does not stack the per-expert tensors
        # fail here, do not write a wrong GGUF; only --vocab-only works for now
        raise NotImplementedError(f"Kolibri tensor conversion is not implemented yet, use --vocab-only ({name})")
