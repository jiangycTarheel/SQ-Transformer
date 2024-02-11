from typing import Dict, List, Optional, Tuple
import math

from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer,
    MultiheadAttention,
    TransformerEncoderLayer,
    LayerNorm
)
import torch
from torch import Tensor
from fairseq import utils


from nat.layer import BlockedEncoderLayer, BlockedDecoderLayer
from nat.modules import FeedForward, RelativeSelfAttention, JYCMultiheadAttention

'''
Author: Yichen Jiang
-- This file implements the shadow attention (SALRelativeSelfAttention), SAL for encoder (SALBlockedEncoderLayer),
and SAL for decoder (SALBlockedDecoderLayer)

-- The (contextualized) quantized word embeddings (codes) are called `proto_x' (short for prototype of x)

'''
class SALRelativeSelfAttention(RelativeSelfAttention):
    """Multi-headed attention with relative attentions.

    See "Self Attention with relative positions" for more details.
    """

    def forward(
            self,
            proto_query,
            proto_key: Optional[Tensor],
            proto_value: Optional[Tensor],
            value: Optional[Tensor],
            pos_key=None,
            pos_val=None,
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            overwrite_incremental_state=True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = proto_query.size()
        assert embed_dim == self.embed_dim,(embed_dim, self.embed_dim)
        assert list(proto_query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    proto_key = proto_value = value = None
        else:
            saved_state = None

        # self-attention
        if self.self_attention:
            v = self.v_proj(value)
            proto_q_full = self.q_proj(proto_query)
            proto_k = self.k_proj(proto_query)
            proto_v = self.v_proj(proto_query)
        else:
            assert proto_key is not None and proto_value is not None
            v = self.v_proj(value)
            proto_q_full = self.q_proj(proto_query)
            proto_k = self.k_proj(proto_key)
            proto_v = self.v_proj(proto_value)

        proto_q_full *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            # k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            # v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )
            proto_k = torch.cat([proto_k, self.bias_k.repeat(1, bsz, 1)])
            proto_v = torch.cat([proto_v, self.bias_v.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        proto_q = proto_q_full.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if proto_k is not None:
            proto_k = (proto_k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if proto_v is not None:
            proto_v = proto_v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_proto_key = saved_state["prev_key"]
                assert _prev_proto_key is not None
                _prev_proto_key = _prev_proto_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    proto_k = _prev_proto_key
                else:
                    assert proto_k is not None
                    proto_k = torch.cat([_prev_proto_key, proto_k], dim=1)

            if "prev_value" in saved_state:
                _prev_proto_value = saved_state["prev_value"]
                assert _prev_proto_value is not None
                prev_proto_value = _prev_proto_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    proto_v = prev_proto_value
                else:
                    assert proto_v is not None
                    proto_v = torch.cat([prev_proto_value, proto_v], dim=1)

            if "prev_value_nonproto" in saved_state:
                _prev_value = saved_state["prev_value_nonproto"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert proto_k is not None and proto_v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=proto_k.size(1),
                static_kv=static_kv,
            )

            if overwrite_incremental_state:
                saved_state["prev_key"] = proto_k.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = proto_v.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value_nonproto"] = v.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_key_padding_mask"] = key_padding_mask
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
            # In this branch incremental_state is never None
            assert incremental_state is not None

        assert proto_k is not None
        src_len = proto_k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, (bsz, key_padding_mask.size(0))
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            proto_k = torch.cat([proto_k, proto_k.new_zeros((proto_k.size(0), 1) + proto_k.size()[2:])], dim=1)
            proto_v = torch.cat([proto_v, proto_v.new_zeros((proto_v.size(0), 1) + proto_v.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            # if k is not None:
            #     k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = self.relative_attention(
            proto_q.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            proto_k.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_key,
        ).contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            raise NotImplementedError
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert proto_v is not None and v is not None
        proto_attn = self.relative_combine(
            probs=attn_probs.contiguous().view(bsz, self.num_heads, tgt_len, src_len),
            value=proto_v.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_val=pos_val
        ).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        attn = self.relative_combine(
            probs=attn_probs.contiguous().view(bsz, self.num_heads, tgt_len, src_len),
            value=v.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_val=pos_val
        ).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
            proto_attn = proto_attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            proto_attn = proto_attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        proto_attn = self.out_proj(proto_attn)
        attn_weights: Optional[Tensor] = None

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, proto_attn, (proto_q_full, attn_weights)


class SALBlockedEncoderLayer(BlockedEncoderLayer):
    def __init__(self, args, relative_keys=None, relative_vals=None):
        super().__init__(args)
        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.enc_block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

    def build_self_attention(self, embed_dim, args):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            raise NotImplementedError
        else:
            return SALRelativeSelfAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward(self, x, proto_x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)

        residual = x
        proto_residual = proto_x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.relative_keys is None:
            x, proto_x, attn = self.self_attn(
                value=x,
                proto_query=proto_x,
                proto_key=proto_x,
                proto_value=proto_x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            index = utils.new_arange(x, x.size(0))
            pos_key_embed = self.relative_keys(index)
            pos_val_embed = self.relative_vals(index)

            x, proto_x, (q, attn) = self.self_attn(
                # query=x,
                # key=x,
                value=x,
                proto_query=proto_x,
                proto_key=proto_x,
                proto_value=proto_x,
                pos_key=pos_key_embed,
                pos_val=pos_val_embed,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        ### Prototype x
        proto_x = self.dropout_module(proto_x)
        proto_x = proto_residual + proto_x
        if not self.normalize_before:
            proto_x = self.self_attn_layer_norm(proto_x)

        proto_residual = proto_x
        if self.normalize_before:
            proto_x = self.final_layer_norm(proto_x)

        proto_x = self.activation_fn(self.fc1(proto_x))
        proto_x = self.activation_dropout_module(proto_x)
        proto_x = self.fc2(proto_x)
        proto_x = self.dropout_module(proto_x)

        if self.ffn_block is None:
            proto_x = proto_residual + proto_x
        else:
            proto_g = self.ffn_block(proto_residual).sigmoid()
            proto_x = proto_residual * proto_g + proto_x * (1 - proto_g)

        if not self.normalize_before:
            proto_x = self.final_layer_norm(proto_x)
        return x, proto_x, {'q': q, 'attn': attn}


class SALBlockedDecoderLayer(BlockedDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
        relative_keys=None, relative_vals=None
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals
        if getattr(args, "decoder_no_self_attn", False):
            self.self_attn = None
        else:
            self.self_attn = self.build_self_attention(
                self.embed_dim,
                args,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
            )
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        if self.encoder_attn_layer_norm is not None:
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

        self.encoder_attn_v = getattr(args, "encoder_attn_v", "encoder_out")
        self.encoder_attn_k = getattr(args, "encoder_attn_k", "encoder_out")

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        if getattr(args, "self_attn_cls", "abs") == "abs":
            raise NotImplementedError
        else:
            return SALRelativeSelfAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def build_encoder_attention(self, embed_dim, args):
        return JYCMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(
            self,
            x,
            proto_x,
            encoder_attn_key: Optional[torch.Tensor] = None,
            encoder_attn_value: Optional[torch.Tensor] = None,
            encoder_attn_proto_key: Optional[torch.Tensor] = None,
            encoder_attn_proto_value: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            overwrite_self_attn_state=True,
            overwrite_cross_attn_state=True
    ):

        if need_head_weights:
            need_attn = True

        residual = x
        proto_residual = proto_x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            proto_x = self.self_attn_layer_norm(proto_x)
        if prev_self_attn_state is not None:
            prev_key, prev_value, prev_value_nonproto = prev_self_attn_state[:3]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
                "prev_value_nonproto": prev_value_nonproto,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
            raise NotImplementedError ## _set_input_buffer here is not compatible with overwrite_self_attn_state.

        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state) if self.self_attn is not None else None

        proto_k = proto_x
        proto_v = proto_x
        v = x
        k = x

        if self.self_attn is not None:
            if self.relative_keys is None:
                x, proto_x, attn = self.self_attn(
                    proto_query=proto_x,
                    proto_key=proto_k,
                    proto_value=proto_v,
                    value=v,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=False,
                    attn_mask=self_attn_mask,
                )
            else:
                if incremental_state is not None:
                    # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                    saved_state = self.self_attn._get_input_buffer(incremental_state)

                    if "prev_key" in saved_state:
                        _prev_key = saved_state["prev_key"]
                        assert _prev_key is not None
                        bsz = proto_k.size(1)
                        prev_key = _prev_key.view(bsz * self.self_attn.num_heads, -1, self.self_attn.head_dim)
                        # if static_kv:
                        #     tmp_k = prev_key
                        # else:
                        assert proto_k is not None
                        # print(prev_key.size())
                        # print(k.size())
                        # tmp_k = torch.cat([prev_key, k], dim=1)
                        index = utils.new_arange(proto_x, prev_key.size(1) + proto_k.size(0))
                    else:
                        index = utils.new_arange(proto_x, proto_x.size(0))
                else:
                    index = utils.new_arange(x, x.size(0))
                pos_key_embed = self.relative_keys(index)
                pos_val_embed = self.relative_vals(index)

                x, proto_x, attn = self.self_attn(
                    proto_query=proto_x,
                    proto_key=proto_k,
                    proto_value=proto_v,
                    value=v,
                    pos_key=pos_key_embed,
                    pos_val=pos_val_embed,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=False,
                    attn_mask=self_attn_mask,
                    overwrite_incremental_state=overwrite_self_attn_state
                )

            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            proto_x = self.dropout_module(proto_x)
            proto_x = proto_residual + proto_x
            if not self.normalize_before:
                proto_x = self.self_attn_layer_norm(proto_x)

        if self.encoder_attn is not None:
            residual = x
            proto_residual = proto_x
            if self.normalize_before:
                # x = self.encoder_attn_layer_norm(x)
                proto_x = self.encoder_attn_layer_norm(proto_x)
            if prev_attn_state is not None:
                prev_key, prev_value, prev_value_nonproto = prev_attn_state[:3]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                    "prev_value_nonproto": prev_value_nonproto,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                print('set encoder_attn buffer')
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            if self.encoder_attn_v == 'proto_encoder_out':
                assert False, "Have to attend to the encoder's outputs calculated using word embeddings (instead of code embeddings)."

            crossattn_output, attn = self.encoder_attn(
                query=proto_x,
                key=encoder_attn_proto_key if self.encoder_attn_k == 'proto_encoder_out' else encoder_attn_key,
                value=encoder_attn_value,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                overwrite_incremental_state=overwrite_cross_attn_state
            )
            crossattn_output = self.dropout_module(crossattn_output)

            x = residual + crossattn_output
            proto_x = proto_residual + crossattn_output

            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
                proto_x = self.encoder_attn_layer_norm(proto_x)

        residual = x
        proto_residual = proto_x
        if self.normalize_before:
            x = self.final_layer_norm(x)
            proto_x = self.final_layer_norm(proto_x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        proto_x = self.activation_fn(self.fc1(proto_x))
        proto_x = self.activation_dropout_module(proto_x)
        proto_x = self.fc2(proto_x)
        proto_x = self.dropout_module(proto_x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.ffn_block is None:
            proto_x = proto_residual + proto_x
        else:
            proto_g = self.ffn_block(proto_residual).sigmoid()
            proto_x = proto_residual * proto_g + proto_x * (1 - proto_g)

        if not self.normalize_before:
            proto_x = self.final_layer_norm(proto_x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_value_nonproto"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"], saved_state["prev_value_nonproto"]]
            return x, proto_x, attn, self_attn_state
        return x, proto_x, attn, None
