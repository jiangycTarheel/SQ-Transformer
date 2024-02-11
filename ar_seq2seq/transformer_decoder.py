import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, NamedTuple
import torch.nn.functional as F

from fairseq.models.transformer import TransformerDecoder
from nat.layer import BlockedDecoderLayer

from .transformer_encoder import EncoderOut, build_relative_embeddings

class TransformerDecoder4Parsing(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.mask_input = getattr(args, "mask_decoder_input", None)
        self.encoder_attn_k = getattr(args, "encoder_attn_k", "encoder_out")
        self.encoder_attn_v = getattr(args, "encoder_attn_v", "encoder_out")

        if getattr(args, "self_attn_cls", "abs") != "abs":
            self.embed_positions = None  # TODO check
            rel_keys = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            rel_vals = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            self.layers = torch.nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_decoder_layer(args, _il, no_encoder_attn, rel_keys, rel_vals)
                    for _il in range(args.decoder_layers)
                ]
            )

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    @staticmethod
    def add_args(parser):
        parser.add_argument("--encoder-attn-k", type=str,
                            choices=["encoder_out", "enc_vq_code", "proto_encoder_out", "encoder_layer_out"],
                            default="encoder_out")
        parser.add_argument("--encoder-attn-v", type=str,
                            choices=["encoder_out", "enc_vq_code", "proto_encoder_out", "encoder_layer_out"],
                            default="encoder_out")
        parser.add_argument("--dec-no-bias-for-cross-attn", action="store_true")
        parser.add_argument("--dec-no-bias-for-self-attn", action="store_true")

    def build_decoder_layer(self, args, ilayer=-1, no_encoder_attn=False, rel_keys=None, rel_vals=None):
        if getattr(args, "block_cls", "None") == "highway" or getattr(args, "dec_self_attn_cls", "abs") != "abs":
            if getattr(args, "self_attn_cls", "abs") == "abs":
                return BlockedDecoderLayer(args, no_encoder_attn)
            else:
                return BlockedDecoderLayer(
                    args, no_encoder_attn=no_encoder_attn,
                    relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                    relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
                )

        return super().build_decoder_layer(args, no_encoder_attn)

    def forward_as_submodule(
        self,
        inputs,
        self_attn_padding_mask,
        encoder_out=None,
        normalize=False,
        tgt_tokens=None,
        incremental_state=None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **unused
    ):
        """ can be extended as extract_features(), w/ forward decoder inputs & w/o embedding projection """
        x = inputs

        features, ret = self._forward_decoding(
            x,
            self_attn_padding_mask,
            encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
        )

        ret["ref_mask"] = self_attn_padding_mask
        ret["inputs"] = x
        ret["features"] = features
        decoder_out = self.output_layer(features)

        if ret.get("predict", None) is None:
            ret["predict"] = decoder_out.max(dim=-1)

        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out

        if tgt_tokens is not None:
            return decoder_out, ret
        else:
            return decoder_out

    def _forward_decoding(self,
        x,
        self_attn_padding_mask,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        overwrite_self_attn_state=True,
        overwrite_cross_attn_state=True,
        debug=False,
    ):

        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        if encoder_out is not None:
            if self.encoder_attn_k == "encoder_out":
                encoder_attn_k = encoder_out.encoder_out
            elif self.encoder_attn_k == "enc_vq_code":
                encoder_attn_k = encoder_out.vq_code
            elif self.encoder_attn_k == "encoder_layer_out":
                encoder_attn_k = encoder_out.encoder_states
            else:
                raise NotImplementedError

            if self.encoder_attn_v == "encoder_out":
                encoder_attn_v = encoder_out.encoder_out
            elif self.encoder_attn_v == "enc_vq_code":
                encoder_attn_v = encoder_out.vq_code
            elif self.encoder_attn_v == "encoder_layer_out":
                encoder_attn_v = encoder_out.encoder_states
            else:
                raise NotImplementedError

            encoder_attn_mask = encoder_out.encoder_padding_mask
        else:
            encoder_attn_k, encoder_attn_v, encoder_attn_mask = None, None, None

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if encoder_out is not None:
                if self.encoder_attn_k == "encoder_layer_out":
                    encoder_attn_k = encoder_out.encoder_states[idx]
                if self.encoder_attn_v == "encoder_layer_out":
                    encoder_attn_v = encoder_out.encoder_states[idx]

            x, layer_attn, _ = layer(
                x,
                encoder_attn_k,
                encoder_attn_v,
                encoder_attn_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                overwrite_self_attn_state=overwrite_self_attn_state,
                overwrite_cross_attn_state=overwrite_cross_attn_state,
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                self_attn, cross_attn = layer_attn['self_attn'], layer_attn['cross_attn']
                if cross_attn is not None:
                    attn = cross_attn.float().to(x)
                else:
                    attn = None


        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, incremental_state, add_position=False, from_inference_z=False,
                          token_embedding: Optional[torch.Tensor] = None):
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None and positions is not None:
            positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(prev_output_tokens)

        x = self.embed_scale * token_embedding

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None and add_position:
            x += positions

        x = self.dropout_module(x)
        if prev_output_tokens is not None:
            decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        else:
            decoder_padding_mask = None
        return x, decoder_padding_mask

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if self.mask_input == 'random' and self.training:
            bsz, dec_seq_len = prev_output_tokens.size(0), prev_output_tokens.size(1)
            rand_num = torch.rand(bsz).cuda()
            random_mask = torch.cuda.FloatTensor(bsz, dec_seq_len).uniform_() > rand_num.unsqueeze(1)
            prev_output_tokens = prev_output_tokens.masked_fill_(random_mask, self.trg_vocab_size - 1)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.mask_input == 'all':
            x = torch.zeros_like(x).cuda()

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        features, ret = self._forward_decoding(
            x,
            self_attn_padding_mask,
            encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        return features, ret