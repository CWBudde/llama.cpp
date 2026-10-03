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
        # the base class writes block count, embedding length, head counts, key/value length and rms eps
        super().set_gguf_parameters()
        # the sliding-window layers rotate the full head
        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the generic TensorNameMap maps the sandwich norms to wrong tensors and does not stack the per-expert tensors
        # fail here, do not write a wrong GGUF
        raise NotImplementedError(f"Kolibri tensor conversion is not implemented yet ({name})")
