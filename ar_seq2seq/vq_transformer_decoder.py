from typing import Any, Dict, List, Optional, Tuple
import copy

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.fairseq_encoder import EncoderOut
from .transformer_encoder import build_relative_embeddings
from .transformer_decoder import TransformerDecoder4Parsing
from .vector_quantization import VectorQuantize
from .utils import Dictionary
from .sal_layer import SALBlockedDecoderLayer
from nat.layer import BlockedDecoderLayer

class VQTransformerDecoder4Parsing(TransformerDecoder4Parsing):

    def build_decoder_layer(self, args, ilayer=None, no_encoder_attn=False, rel_keys=None, rel_vals=None):

        if getattr(args, "dec_vq_use_shadow_attn", False):
            decoder_layer_cls = SALBlockedDecoderLayer
        else:
            decoder_layer_cls = BlockedDecoderLayer
        if getattr(args, "block_cls", "None") == "highway" or getattr(args, "dec_self_attn_cls", "abs") != "abs":
            if getattr(args, "self_attn_cls", "abs") == "abs":
                return decoder_layer_cls(args, no_encoder_attn)
            else:
                return decoder_layer_cls(
                    args, no_encoder_attn=no_encoder_attn,
                    relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                    relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
                )

        return super().build_decoder_layer(args, no_encoder_attn)

    def build_vq(self, args, codebook_dim, codebook_size):

        vq = VectorQuantize(
            dim=args.decoder_embed_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,  # codebook size
            decay=args.lamda,
            # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=0.,  # the weight on the commitment loss
            use_cosine_sim=True,
            learnable_codebook=not getattr(args, 'vq_unlearnable_codebook', False),
            xtra_pad_code=getattr(args, "xtra_pad_code", False),
            straightthru_to_soft_code=getattr(args, "vq_straightthru_to_soft_code", False),
            no_straightthru=getattr(args, "vq_no_straightthru", False)
        )
        return vq

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, vq=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)

        self.no_encoder_attn = no_encoder_attn

        vq_codebook_dim = getattr(args, "vq_codebook_dim")
        if vq_codebook_dim is None:
            vq_codebook_dim = args.decoder_embed_dim
        if vq is None:
            self.vq = self.build_vq(args, vq_codebook_dim, args.num_codes)
        else:
            self.vq = vq

        self.latent_use = getattr(args, "latent_use", "input")
        self.latent_factor = getattr(args, "latent_factor", 0.5)

        self.vq_input = args.vq_input

        self.predictor: TransformerDecoder4Parsing = self._build_predictor(args, self.vq)

        self.no_embedding_dropout = getattr(args, "vq_no_embedding_dropout", False)
        self.no_embedding_dropout4_inference_z = getattr(args, "vq_no_embedding_dropout4inference_z", False)

        self.xtra_pad_code = getattr(args, "xtra_pad_code", False)
        self.xtra_eos_code = getattr(args, "xtra_eos_code", False)

        self.decode_proto_x = getattr(args, "dec_vq_decode_proto_x", False)
        self.use_shadow_attn = getattr(args, "dec_vq_use_shadow_attn", False)

        self.dont_update_code_elsewhere = getattr(args, "dec_dont_update_code_elsewhere", False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--latent-dim", type=int, default=200)
        parser.add_argument("--latent-layers", type=int)

        '''
        Arg "latent-use" defines how to 'use' the latent variable z (code):
            input: Only quantize the src embeddings (input), and z is the codes of the tgt_tokens
            input_inference: Only quantize the src embeddings (input), and z is the codes of the prev_output_tokens
            input_inference&output: After all attention layers, also use the contextualized word representations to predict the next code
            input_inference&proto_output: After all attention layers, also use the contextualized code representations (proto_output) to predict the next code
        '''
        parser.add_argument("--latent-use", type=str, default="input",
                            choices=["input",
                                     "input_inference", "input_inference&predict_output", "input_inference&predict_proto_output"])
        parser.add_argument("--latent-factor", type=float)

        # for vector quantization
        parser.add_argument("--num-codes", type=int)
        parser.add_argument("--lamda", type=float, default=0.999)

        parser.add_argument("--vq-dropout", type=float)
        parser.add_argument("--share-bottom-layers", action="store_true", help="pursue a less memory-cost model")

        parser.add_argument("--vq-input", type=str, default="prev_target",
                            choices=["prev_target", "prev_embed", "prev_vq_target_straightthru"])
        parser.add_argument("--vq-predictor-encoder-attn-k", type=str, choices=["encoder_out", "enc_vq_code"],
                            default="encoder_out")
        parser.add_argument("--vq-predictor-encoder-attn-v", type=str, choices=["encoder_out", "enc_vq_code"],
                            default="encoder_out")
        parser.add_argument("--vq-dont-share-code-and-predictor-embed", action="store_true")
        parser.add_argument("--vq-no-embedding-dropout", action="store_true")
        parser.add_argument("--vq-no-cross-attention", action="store_true")
        parser.add_argument("--vq-no-embedding-dropout4inference-z", action="store_true")

        ## Options for vector_quantize_pytorch package
        parser.add_argument('--vq-codebook-dim', type=int, default=None)
        parser.add_argument('--xtra-pad-code', action="store_true")
        parser.add_argument('--vq-straightthru-to-soft-code', action="store_true")
        parser.add_argument('--vq-no-straightthru', action="store_true")
        parser.add_argument('--vq-unlearnable-codebook', action="store_true")


        parser.add_argument("--dec-vq-use-shadow-attn", action="store_true")

        parser.add_argument("--dec-vq-decode-proto-x", action="store_true")
        parser.add_argument("--dec-dont-update-code-elsewhere", action="store_true")

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

        if not self.no_embedding_dropout and not (from_inference_z and self.no_embedding_dropout4_inference_z):
            x = self.dropout_module(x)
        if prev_output_tokens is not None:
            decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        else:
            decoder_padding_mask = None
        return x, decoder_padding_mask

    def forward(
            self,
            prev_output_tokens,
            prev_vq_tokens=None,
            tgt_tokens = None,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            overwrite_self_attn_state=True,
            overwrite_cross_attn_state=True,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            prev_vq_tokens=prev_vq_tokens,
            tgt_tokens=tgt_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            overwrite_self_attn_state=overwrite_self_attn_state,
            overwrite_cross_attn_state=overwrite_cross_attn_state,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def reorder_incremental_state_scripting(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            dec_new_order: Tensor,
            vq_predictor_new_order: Tensor = None,
    ):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        idx = 0
        ## CHANGE
        if vq_predictor_new_order is None:
            vq_predictor_new_order = dec_new_order

        for module in self.predictor.modules():
            idx += 1
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, vq_predictor_new_order)

                if result is not None:
                    incremental_state = result

        for module in self.layers.modules():
            idx += 1
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, dec_new_order)

                if result is not None:
                    incremental_state = result


    def _forward_decoding(
        self,
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
        proto_x=None,
        return_proto_x=False,
        decode_proto_x=True,
    ):
        x = x.transpose(0, 1)
        if proto_x is not None:
            proto_x = proto_x.transpose(0, 1)
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        encoder_attn_proto_v, encoder_attn_proto_k = None, None
        if encoder_out is not None:
            if self.encoder_attn_k == "encoder_out":
                encoder_attn_k = encoder_out.encoder_out
                encoder_attn_proto_k = encoder_out.encoder_out
            elif self.encoder_attn_k == "enc_vq_code":
                encoder_attn_k = encoder_out.vq_code
            elif self.encoder_attn_k == "proto_encoder_out":
                encoder_attn_k = encoder_out.encoder_out
                encoder_attn_proto_k = encoder_out.proto_encoder_out
            elif self.encoder_attn_k == "encoder_layer_out":
                encoder_attn_k = encoder_out.encoder_states
            else:
                raise NotImplementedError

            if self.encoder_attn_v == "encoder_out":
                encoder_attn_v = encoder_out.encoder_out
                encoder_attn_proto_v = encoder_out.encoder_out
            elif self.encoder_attn_v == "enc_vq_code":
                encoder_attn_v = encoder_out.vq_code
            elif self.encoder_attn_v == "proto_encoder_out":
                encoder_attn_v = encoder_out.encoder_out
                encoder_attn_proto_v = encoder_out.proto_encoder_out
            elif self.encoder_attn_v == "encoder_layer_out":
                encoder_attn_v = encoder_out.encoder_states
            else:
                raise NotImplementedError

            encoder_attn_mask = encoder_out.encoder_padding_mask
        else:
            encoder_attn_k, encoder_attn_v, encoder_attn_mask = None, None, None
        proto_x_ret = {}

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

            if self.use_shadow_attn:
                x, proto_x, layer_attn, _ = layer(
                    x,
                    proto_x,
                    encoder_attn_k,
                    encoder_attn_v,
                    encoder_attn_proto_k,
                    encoder_attn_proto_v,
                    encoder_attn_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    overwrite_self_attn_state=overwrite_self_attn_state,
                    overwrite_cross_attn_state=overwrite_cross_attn_state,
                )

                proto_x_ret[f'x_proto_x_similarity_{idx}'] = {
                    'z_q_st': proto_x.transpose(0, 1),
                    'z_e': x.transpose(0, 1)
                }
            else:
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
                if self.decode_proto_x and decode_proto_x:
                    proto_x, proto_layer_attn, _ = layer(
                        proto_x,
                        encoder_attn_proto_k,
                        encoder_attn_proto_v,
                        encoder_attn_mask,
                        incremental_state,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=self_attn_padding_mask,
                        need_attn=bool((idx == alignment_layer)),
                        need_head_weights=bool((idx == alignment_layer)),
                        overwrite_self_attn_state=overwrite_self_attn_state,
                        overwrite_cross_attn_state=overwrite_cross_attn_state,
                    )

                    """
                    JYC: Use the code below if you want to visualize the attention weights
                    """
                    # self_attn, cross_attn = layer_attn['self_attn'], layer_attn['cross_attn']  # [bsz, num_heads, seq_len, seq_len]
                    # proto_self_attn, proto_cross_attn = proto_layer_attn['self_attn'], proto_layer_attn['cross_attn']  # [bsz, num_heads, seq_len, seq_len]
                    #
                    # proto_x_ret[f'x_proto_x_self_attn_weights_l{idx}'] = {
                    #     'proto_attn': proto_self_attn,
                    #     'attn': self_attn
                    # }
                    #
                    # proto_x_ret[f'x_proto_x_cross_attn_weights_l{idx}'] = {
                    #     'proto_attn': proto_cross_attn,
                    #     'attn': cross_attn
                    # }

                    proto_x_ret[f'x_proto_x_similarity_{idx}'] = {
                        'z_q_st': proto_x.transpose(0, 1),
                        'z_e': x.transpose(0, 1)
                    }

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

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

        if proto_x is not None:
            if self.layer_norm is not None:
                proto_x = self.layer_norm(proto_x)

            # T x B x C -> B x T x C
            proto_x = proto_x.transpose(0, 1)

            if self.project_out_dim is not None:
                proto_x = self.project_out_dim(proto_x)
        if return_proto_x:
            return x, proto_x, {"attn": [attn], "inner_states": inner_states, "proto_x_ret": proto_x_ret}
        else:
            return x, {"attn": [attn], "inner_states": inner_states}


    def decode(
            self,
            prev_output_tokens,
            prev_vq_tokens=None,
            tgt_tokens = None,
            encoder_out_for_vq: Optional[EncoderOut] = None,
            encoder_out_for_dec: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            overwrite_self_attn_state=True,
            overwrite_cross_attn_state=True,
            infer_with_tgt=False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if tgt_tokens is not None:
                tgt_tokens = tgt_tokens[:, -1:]

        # embed tokens and positions
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens, incremental_state)
        x_embed = x

        if self.mask_input == 'all':
            x = torch.zeros_like(x).cuda()

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        z_ret = None
        proto_x = None
        if self.latent_use in ["input", "input_inference",
                               "input_inference&predict_output", "input_inference&predict_proto_output"]:

            vq_input = self.build_vq_input(x, prev_output_tokens, incremental_state,
                                           vq_input_type=self.vq_input, prev_vq_tokens=prev_vq_tokens, vq=self.vq)

            if infer_with_tgt and not self.training and tgt_tokens is not None:
                z = self._inference_z(None, decoder_padding_mask, incremental_state, tgt_tokens, vq=self.vq)[0]["z_q_st"]
                z_ret = None
            else:
                z, z_ret = self.decode_z(
                    encoder_out_for_vq,
                    incremental_state=incremental_state,
                    inputs=vq_input,
                    decoder_padding_mask=decoder_padding_mask,
                    tgt_tokens=tgt_tokens,
                )

            ### Unless latent_use == 'input', otherwise z is the code of prev_output_tokens instead of tgt_tokens
            if self.latent_use.startswith("input_inference"):
                z = vq_input

            proto_x = z

        _bsz, _t, _c = x.size()

        residual = x

        x = residual

        features, ret = self._forward_decoding(
            x,
            decoder_padding_mask,
            encoder_out_for_dec,
            incremental_state,
            overwrite_self_attn_state=overwrite_self_attn_state,
            overwrite_cross_attn_state=overwrite_cross_attn_state,
            proto_x=proto_x,
            decode_proto_x=False
        )

        ret["inputs"] = residual
        ret["features"] = features
        if z_ret is not None:
            ret.update(z_ret)

        if not features_only:
            features = self.output_layer(features)

        return features, ret

    def decode_z(self, encoder_out, incremental_state, inputs=None, decoder_padding_mask=None,
                 tgt_tokens=None, predict_padding_mask=None, left_shift_vq_target=False):

        if tgt_tokens is not None:
            inference_out, posterior_idx = self._inference_z(None, decoder_padding_mask, incremental_state, tgt_tokens,
                                                             vq=self.vq, update_code=False)
        else:
            posterior_idx = None


        if self.predictor is None:
            # a non-parameterize predictor, nearest search with decoder inputs
            predict_out, idx = self._inference_z(inputs, decoder_padding_mask, incremental_state, vq=self.vq)

        else:
            # a parameterize predictor, we use GLAT here.
            ret = self.predictor.forward_as_submodule(
                inputs,
                decoder_padding_mask,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                tgt_tokens=None,
            )

            z_out, ext = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, {})

            z_out = F.log_softmax(z_out, dim=-1)
            z_scores, z_idx = torch.topk(z_out, 1, dim=-1)

            z_idx = z_idx.view(z_idx.size(0) * 1, 1)
            q = self.forward_code(z_idx, vq=self.vq) #self.code.forward(indices=z_idx)

            return q, \
                   {'vq':[{
                        "idx": {
                            'prior': z_idx,
                            'posterior': posterior_idx
                        },
                        'lprobs': z_scores,
                        }]
                }

    def extract_features(
            self,
            prev_output_tokens,
            prev_vq_tokens=None,
            tgt_tokens=None,
            encoder_out: Optional[EncoderOut] = None,
            structure_encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            overwrite_self_attn_state=True,
            overwrite_cross_attn_state=True,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            prev_vq_tokens,
            tgt_tokens,
            encoder_out,
            structure_encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            overwrite_self_attn_state=overwrite_self_attn_state,
            overwrite_cross_attn_state=overwrite_cross_attn_state,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens=None,
        prev_vq_tokens=None,
        tgt_tokens=None,
        encoder_out: Optional[EncoderOut] = None,
        structure_encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        overwrite_self_attn_state=True,
        overwrite_cross_attn_state=True,
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

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if tgt_tokens is not None:
                tgt_tokens = tgt_tokens[:, -1:]

        # embed tokens and positions
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens, incremental_state)
        x_embed = x

        if self.mask_input == 'all':
            x = torch.zeros_like(x).cuda()

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        latent_encoder_out = encoder_out

        proto_x = None

        if self.latent_use in ["input_inference",
                               "input_inference&predict_output", "input_inference&predict_proto_output"]:
            # Input to the VQ predictor
            vq_input_x = x
            vq_input = self.build_vq_input(vq_input_x, prev_output_tokens, incremental_state,
                                           vq_input_type=self.vq_input, prev_vq_tokens=prev_vq_tokens, vq=self.vq)
            z, z_ret = self.forward_z(
                latent_encoder_out,
                incremental_state,
                tgt_tokens=tgt_tokens,
                inputs=vq_input,
                decoder_padding_mask=decoder_padding_mask,
                vq=self.vq,
                predictor=self.predictor
            )

            if self.latent_use.startswith("input_inference"):
                z = vq_input

            proto_x = z

        residual = x  # x: B, T, C

        x = residual

        features, proto_features, ret = self._forward_decoding(
            x,
            decoder_padding_mask,
            encoder_out,
            incremental_state,
            overwrite_self_attn_state=overwrite_self_attn_state,
            overwrite_cross_attn_state=overwrite_cross_attn_state,
            proto_x=proto_x,
            return_proto_x=True
        )
        if self.decode_proto_x:
            proto_x_ret = ret["proto_x_ret"]
            z_ret.update(proto_x_ret)

        ### Finally, after all attention layers, predict the code of the next token

        if self.latent_use in ['input_inference&predict_output', 'input_inference&predict_proto_output']:
            out_vq_src = proto_features if self.latent_use == 'input_inference&predict_proto_output' else features
            output_z, output_z_idx, _, output_z_soft_posterior = self.quantize(
                out_vq_src,
                prev_output_tokens,
                vq=self.vq,
                update_code=False
            )

            output_z_tgt = z_ret['posterior']['idx'] if tgt_tokens is not None else None

            output_z_ret = {
                "output_VQ": {
                    "posterior": {
                        "z_q_st": output_z,
                        "z_e": out_vq_src
                    },
                    "out": output_z_soft_posterior.view(out_vq_src.size(0), out_vq_src.size(1), output_z_soft_posterior.size(-1)),
                    "tgt": output_z_tgt,
                    "factor": self.latent_factor,
                    "mask": ~decoder_padding_mask
                }
            }
            ret.update(output_z_ret)

        ret["inputs"] = residual
        ret["features"] = features
        if z_ret is not None:
            ret.update(z_ret)
        return features, ret

    def quantize(self, inputs, inputs_idx, mask=None, vq=None, update_code=True):
        quantized, vq_idx, loss, dist = vq(inputs, x_idx=inputs_idx, mask=mask, update_code=update_code)
        return quantized, vq_idx, loss, dist

    def forward_code(self, idx, vq_input_code=None, vq=None, detach_code=True):
        if vq_input_code is not None:
            q = vq_input_code.forward(indices=idx)
        else:
            q = vq.forward_code(idx.unsqueeze(0), detach_code=detach_code).squeeze(0)

        return q

    def build_vq_input_v2(self, vq_input_x, prev_output_tokens, incremental_state, x_embed, prev_vq_tokens=None, vq=None):

        vq_input_code = None

        if self.vq_input == "prev_target":
            vq_input = vq_input_x
        elif self.vq_input == "prev_embed":
            vq_input = x_embed
        elif self.vq_input.startswith("prev_vq_target"):
            if prev_vq_tokens is None:
                vq_input, vq_idx, _, _ = self.quantize(
                    vq_input_x,
                    inputs_idx=prev_output_tokens,
                    vq=vq,
                    update_code=not self.dont_update_code_elsewhere
                )

            else:
                vq_input = self.forward_code(prev_vq_tokens, vq_input_code=vq_input_code, vq=vq) # vq_input_code.forward(indices=prev_vq_tokens)
        else:
            raise NotImplementedError

        return vq_input

    def build_vq_input(self, vq_input_x, prev_output_tokens, incremental_state, vq, vq_input_type, prev_vq_tokens=None):

        vq_input_code = None

        if vq_input_type == "prev_target":
            vq_input = vq_input_x
        elif vq_input_type.startswith("prev_vq_target"):
            if prev_vq_tokens is None:
                inputs = self.forward_embedding(
                    prev_output_tokens,
                    incremental_state,
                    add_position=False,
                    from_inference_z=True,
                )[0]
                vq_input, vq_idx, _, _ = self.quantize(
                    inputs,
                    inputs_idx=prev_output_tokens,
                    vq=vq,
                    update_code=not self.dont_update_code_elsewhere
                )
            else:
                vq_input = self.forward_code(prev_vq_tokens, vq_input_code=vq_input_code, vq=vq)
        else:
            raise NotImplementedError

        return vq_input

    def forward_z(self, encoder_out, incremental_state, tgt_tokens=None, inputs=None,
                  decoder_padding_mask=None, predict_padding_mask=None, left_shift_vq_target=False,
                  predictor=None, vq=None):
        if predict_padding_mask is None:
            predict_padding_mask = decoder_padding_mask

        if tgt_tokens is not None:
            # vector quantization from the reference --- non-parameter posterior
            inference_out, idx = self._inference_z(None, decoder_padding_mask, incremental_state, tgt_tokens, vq=vq)
        else:
            inference_out, idx = None, None

        if left_shift_vq_target and idx is not None:
            idx = idx[:, 1:].contiguous()
            decoder_padding_mask = decoder_padding_mask[:, 1:].contiguous()

        if self.predictor is None:
            # a non-parameterize predictor, nearest search with decoder inputs
            predict_out, idx = self._inference_z(inputs, decoder_padding_mask, incremental_state, vq=vq)
        else:
            predict_out, idx = self._predict_z(predictor, inputs, predict_padding_mask, encoder_out, incremental_state,
                                               tgt=idx, out=inference_out)

        if inference_out is not None:  ## Training time, q is
            q = inference_out["z_q_st"]
            return q, {"prior": predict_out, "posterior": inference_out, "vq_ret": {"prior_out": idx}}
        else:                          ## Inference/Test time, q is the closest code embedding
            q = self.forward_code(idx, vq=vq) #self.code.forward(indices=idx)
            return q, {"prior": predict_out, "posterior": inference_out, "vq_ret": {"prior_out": idx}}

    def _inference_z(self, inputs, decoder_padding_mask, incremental_state, tgt_tokens=None, vq=None, update_code=True):
        if inputs is None:
            if tgt_tokens is not None:
                # TODO: switch to a context-aware representation, instead of context-independent embeddings
                inputs = self.forward_embedding(
                    tgt_tokens,
                    incremental_state,
                    add_position=False,
                    from_inference_z=True,
                )[0]
            else:
                raise NotImplementedError

        z_q_st, idx, vq_loss, soft_posterior = self.quantize(
            inputs,
            inputs_idx=tgt_tokens,
            mask=~decoder_padding_mask,
            vq=vq,
            update_code=update_code
        )
        z_q = None
        soft_posterior = soft_posterior.view(inputs.size(0), inputs.size(1), soft_posterior.size(2))

        return {
                    "z_q_st": z_q_st,    # vq_st output, detached, no gradient
                    "z_q": z_q,
                    "z_e": inputs,
                    "idx": idx,
                    "mask": ~decoder_padding_mask,
                    "soft_posterior": soft_posterior,
                    "precomputed_loss": vq_loss
               }, idx

    def _predict_z(self, predictor, inputs, decoder_padding_mask, encoder_out, incremental_state, tgt=None, out=None):
        """ predict the latent variables """
        ret = predictor.forward_as_submodule(
            inputs,
            decoder_padding_mask,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            tgt_tokens=tgt,
        )
        z_out, ext = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, {})
        z_idx = z_out.max(dim=-1)[1]

        mask = decoder_padding_mask

        if tgt is None:
            # predict latent variables while testing
            return {}, z_idx
        else:
            # using the approximate ground truth of latent codes
            z_mix = out["z_q_st"]

            return {
               "VQ": {
                   "out": z_out,  # glancing outputs
                   "tgt": tgt,  # reference target
                   "factor": self.latent_factor,
                   "mask": ~(decoder_padding_mask + ext["ref_mask"] > 0.)
               },
               "z_input": z_mix,
               "z_pred": ext.get("predict", None),
               "z_ref": out.get("z_q_st", None),
               "ref_mask": ext["ref_mask"] > 0.
           }, z_idx

    def _build_predictor(self, main_args, vq):
        args = copy.deepcopy(main_args)
        args.share_decoder_input_output_embed = not getattr(main_args, "vq_split", not self.share_input_output_embed)
        args.decoder_layers = getattr(main_args, "latent_layers", main_args.decoder_layers)
        args.decoder_embed_dim = getattr(args, "vq_predictor_embed_dim", args.decoder_embed_dim)
        args.decoder_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", args.decoder_ffn_embed_dim)
        args.decoder_attention_heads = getattr(args, "vq_predictor_attention_heads", args.decoder_attention_heads)
        args.decoder_output_dim = args.decoder_embed_dim
        if not getattr(main_args, "vq_no_cross_attention", False):
            args.encoder_embed_dim = getattr(args, "vq_struct_encoder_embed_dim", args.encoder_embed_dim)

        args.dropout = getattr(main_args, "vq_dropout", main_args.dropout)
        args.encoder_attn_k = getattr(main_args, "vq_predictor_encoder_attn_k", None)
        args.encoder_attn_v = getattr(main_args, "vq_predictor_encoder_attn_v", None)

        args.dec_vq_use_shadow_attn = False

        embed_tokens = torch.nn.Embedding.from_pretrained(
            vq._codebook.embed[0],
            freeze=False,
            padding_idx=-1
        )

        latent_decoder = TransformerDecoder4Parsing(
            args,
            dictionary=Dictionary(num_codes=args.num_codes),
            embed_tokens=embed_tokens,
            no_encoder_attn=getattr(main_args, "vq_no_cross_attention", False)
        )

        del latent_decoder.embed_tokens

        '''
        Had to do this weird trick to share the code embedding and the output_projection, 
        because code embedding has size [num_codebook, num_code_per_codebook, code_dimension].
        In this work, we only use num_codebook=1, so we share the weight of vq._codebook.embed[0] and the output_projection.weight
        '''
        shared_output_projection = myLinear(
            latent_decoder.output_projection.in_features,
            latent_decoder.output_projection.out_features,
            slice=0
        )
        shared_output_projection.weight = vq._codebook.embed
        latent_decoder.output_projection = shared_output_projection
        # assert getattr(main_args, "vq_no_cross_attention", False)

        if getattr(args, "share_bottom_layers", False):
            shared_layers = args.latent_layers if args.decoder_layers > args.latent_layers else args.decoder_layers
            for i in range(shared_layers):
                latent_decoder.layers[i] = self.layers[i]

        return latent_decoder


class myLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, slice: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(myLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.slice = slice

    def forward(self, input):
        return F.linear(input, self.weight[self.slice], self.bias)