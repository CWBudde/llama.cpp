from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

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

        # the expert weights are the sigmoid of the raw router logits
        # they are not renormalized and not scaled
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_weights_scale(1.0)

        # 513 = 512 previous tokens + the current token (LLAMA_SWA_TYPE_STANDARD)
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        is_swa = [t == "sliding_attention" for t in hparams["layer_types"]]
        self.gguf_writer.add_sliding_window_pattern(is_swa)
        # the full-attention layers use no RoPE
        self.gguf_writer.add_rope_pattern(is_swa)

    # the generic TensorNameMap sends post_attention_layernorm and post_attn_norm to the wrong tensors
    # post_attention_layernorm is the pre-FFN norm, as in Qwen; post_attn_norm and post_ffn_norm are the sandwich norms
    _block_tensors = {
        "input_layernorm.weight": (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
        "post_attn_norm.weight": (gguf.MODEL_TENSOR.ATTN_POST_NORM, ".weight"),
        "post_attention_layernorm.weight": (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
        "post_ffn_norm.weight": (gguf.MODEL_TENSOR.FFN_POST_NORM, ".weight"),
        # the routing correction bias, added to the router logits for the top-k selection only
        "moe.router.expert_bias": (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
    }

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if bid is not None:
            local = name.removeprefix(f"model.layers.{bid}.")
            if (mapped := self._block_tensors.get(local)) is not None:
                tensor, suffix = mapped
                yield self.format_tensor_name(tensor, bid, suffix), data_torch
                return
            # mlp.shared_experts takes the generic path below
            if local.startswith("mlp.experts."):
                yield from self._stack_experts(data_torch, name, bid)
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def _stack_experts(self, data_torch: Tensor, name: str, bid: int) -> Iterable[tuple[str, Tensor]]:
        # the checkpoint stores one tensor per expert; llama.cpp wants one 3D tensor per layer and projection
        n_experts = self.hparams["num_experts"]
        if self._experts is None:
            self._experts = [{} for _ in range(self.block_count)]
        self._experts[bid][name] = data_torch
        if len(self._experts[bid]) < n_experts * 3:
            return

        for w_name in ("gate_proj", "up_proj", "down_proj"):
            datas = [self._experts[bid].pop(f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight") for xid in range(n_experts)]
            # the generic map sends the merged name to ffn_{gate,up,down}_exps
            yield from super().modify_tensors(torch.stack(datas, dim=0), f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)

    def prepare_tensors(self) -> None:
        super().prepare_tensors()
        if self._experts is not None and (left := [k for d in self._experts for k in d]):
            raise ValueError(f"Unprocessed experts: {left}")
