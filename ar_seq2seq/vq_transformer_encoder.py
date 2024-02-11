from typing import Any, Dict, List, Optional, NamedTuple
import copy

import torch
from torch import Tensor

from nat.layer import BlockedEncoderLayer
from .transformer_encoder import TransformerEncoder4Parsing, build_relative_embeddings
from .transformer_decoder import TransformerDecoder4Parsing
from .vq_transformer_decoder import myLinear
from .vector_quantization import VectorQuantize
from .utils import Dictionary
from .sal_layer import SALBlockedEncoderLayer

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("vq_code", Tensor), # T x B x C
        ("proto_encoder_out", Optional[Tensor]),  # B x T
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)

class VQTransformerEncoder4Parsing(TransformerEncoder4Parsing):

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None,
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        if not self.no_embedding_dropout:
            x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def build_encoder_layer(self, args, ilayer=None, rel_keys=None, rel_vals=None):
        if getattr(args, "enc_vq_use_shadow_attn", False):
            encoder_layer_cls = SALBlockedEncoderLayer
        else:
            encoder_layer_cls = BlockedEncoderLayer
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return encoder_layer_cls(args)
        else:
            return encoder_layer_cls(
                args,
                relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
            )

    def build_vq(self, args, codebook_dim, codebook_size):

        vq = VectorQuantize(
            dim=args.encoder_embed_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,  # codebook size
            decay=args.enc_lamda,
            # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=0.,  # the weight on the commitment loss
            use_cosine_sim=True,
            learnable_codebook=not getattr(args, 'vq_unlearnable_codebook', False),
            xtra_pad_code=getattr(args, "enc_xtra_pad_code", False),
            straightthru_to_soft_code=getattr(args, "vq_straightthru_to_soft_code", False),
            no_straightthru=getattr(args, "vq_no_straightthru", False)
        )
        return vq

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens)

        vq_codebook_dim = getattr(args, "vq_codebook_dim")
        if vq_codebook_dim is None:
            vq_codebook_dim = args.encoder_embed_dim
        self.vq = self.build_vq(args, vq_codebook_dim, args.enc_num_codes)

        self.latent_use = getattr(args, "enc_latent_use", "input")
        self.latent_factor = getattr(args, "enc_latent_factor", 0.5)

        self.vq_input = args.enc_vq_input

        self.predict_code = getattr(args, "enc_predict_code", False)
        self.predict_masked_code = getattr(args, "enc_predict_masked_code", False)
        if self.predict_code:
            if self.predict_masked_code:
                self.predictor: TransformerEncoder4Parsing = self._build_masked_predictor(args, self.vq)
            else:
                self.predictor: TransformerDecoder4Parsing = self._build_predictor(args, self.vq)
        else:
            self.predictor = None
        self.predict_z_input = args.enc_predict_z_input
        if self.predict_z_input == 'codes' and self.predict_masked_code:
            self.unk_code = torch.nn.Embedding(1, args.decoder_embed_dim)

        self.mask_predictor_input = getattr(args, "vq_mask_encoder_predictor_input", False)
        self.use_shadow_attn = getattr(args, "enc_vq_use_shadow_attn", False)
        self.encode_proto_x = getattr(args, "enc_vq_encode_proto_x", False)
        self.no_embedding_dropout = getattr(args, "enc_vq_no_embedding_dropout", False)

        self.dont_update_code_elsewhere = getattr(args, "enc_dont_update_code_elsewhere", False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--enc-latent-dim", type=int, default=200)
        parser.add_argument("--enc-latent-layers", type=int, default=3)
        parser.add_argument("--enc-latent-use", type=str, default="output",
                            choices=["input", "input&predict_output", "input&predict_proto_output"])
        parser.add_argument("--enc-latent-factor", type=float, default=0.5)

        # for vector quantization
        parser.add_argument("--enc-num-codes", type=int)
        parser.add_argument("--enc-lamda", type=float, default=0.999)
        parser.add_argument("--enc-predict-code", action="store_true")
        parser.add_argument("--enc-predict-masked-code", action="store_true")
        parser.add_argument("--enc-predict-z-input", type=str, default="codes", choices=["codes", "tokens"])

        parser.add_argument("--enc-vq-dropout", type=float)
        parser.add_argument("--enc-share-bottom-layers", action="store_true", help="pursue a less memory-cost model")

        # Newly added arguments for autoregressive models
        parser.add_argument("--enc-vq-input", type=str, default="prev_target",
                            choices=["prev_target", "prev_embed", "prev_vq_target_straightthru"])

        parser.add_argument("--enc-vq-use-shadow-attn", action="store_true")
        parser.add_argument("--enc-vq-no-embedding-dropout", action="store_true")
        parser.add_argument("--enc-vq-encode-proto-x", action="store_true")
        parser.add_argument('--enc-xtra-pad-code', action="store_true")
        parser.add_argument("--enc-dont-update-code-elsewhere", action="store_true")

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # compute padding mask
        all_z_ret = {}
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        proto_x = None

        if self.predict_masked_code:
            vq_input = self.build_vq_input(
                x,
                encoder_embedding,
                self.vq_input,
                self.vq,
                encoder_padding_mask
            )
        else:
            vq_input = self.build_vq_input(
                x[:, :-1],
                encoder_embedding[:, :-1],
                self.vq_input,
                self.vq,
                encoder_padding_mask[:, :-1]
            )
        z, z_ret = self.forward_z(
            predictor=self.predictor,
            vq=self.vq,
            src_tokens=src_tokens,
            inputs=vq_input,
            x=x,
            encoder_padding_mask=encoder_padding_mask
        )
        all_z_ret['z'] = z_ret

        if self.use_shadow_attn or self.encode_proto_x:
            proto_x = z
            proto_x = proto_x.transpose(0, 1)
        z = z.transpose(0, 1).contiguous()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        all_layers_dict = []
        for _il, layer in enumerate(self.layers):
            if self.use_shadow_attn:
                x = layer(
                    x,
                    proto_x,
                    encoder_padding_mask
                )
                if isinstance(x, tuple):
                    x, proto_x, layer_dict = x
                    all_layers_dict.append(layer_dict)
                else:
                    assert False
                all_z_ret[f'x_proto_x_similarity_{_il}'] = {
                    'z_q_st': proto_x.transpose(0, 1),
                    'z_e': x.transpose(0, 1)
                }
            else:

                x = layer(
                    x,
                    encoder_padding_mask,
                    need_head_weights=True
                )
                if isinstance(x, tuple):
                    x, layer_dict = x
                    all_layers_dict.append(layer_dict)

                if self.encode_proto_x:

                    proto_x = layer(
                        proto_x,
                        encoder_padding_mask,
                        need_head_weights=True
                    )

                    if isinstance(proto_x, tuple):
                        proto_x, proto_layer_dict = proto_x
                        attn = layer_dict['attn']  # [bsz, num_heads, seq_len, seq_len]
                        proto_attn = proto_layer_dict['attn']  # [bsz, num_heads, seq_len, seq_len]

                        all_z_ret[f'x_proto_x_attn_weights_l{_il}'] = {
                            'proto_attn': proto_attn,
                            'attn': attn
                        }

                    all_z_ret[f'x_proto_x_similarity_{_il}'] = {
                        'z_q_st': proto_x.transpose(0, 1),
                        'z_e': x.transpose(0, 1)
                    }

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if self.use_shadow_attn or self.encode_proto_x:
                proto_x = self.layer_norm(proto_x)

        if self.latent_use in ['input&predict_output', 'input&predict_proto_output']:
            out_vq_src = proto_x.transpose(0, 1).contiguous() if self.latent_use == 'input&predict_proto_output' else x.transpose(0, 1).contiguous()
            output_z, output_z_idx, _, output_z_soft_posterior = self.quantize(
                out_vq_src,
                src_tokens,
                vq=self.vq,
                update_code=False,
            )
            output_z_tgt = all_z_ret['z']['posterior']['idx']
            output_z_soft_posterior = output_z_soft_posterior.view(
                out_vq_src.size(0), out_vq_src.size(1), output_z_soft_posterior.size(-1)
            )[:, :-1].contiguous()

            output_z_ret = {
                "output_VQ": {
                    "posterior": {
                        "z_q_st": output_z,
                        "z_e": out_vq_src
                    },
                    "out": output_z_soft_posterior,
                    "tgt": output_z_tgt[:, 1:].contiguous(),
                    "factor": self.latent_factor,
                    "mask": ~encoder_padding_mask[:, :-1]
                }
            }

            all_z_ret['z'].update(output_z_ret)

        encoder_out = EncoderOut(
            encoder_out=x,  # T x B x C
            vq_code=z,  # T x B x C
            proto_encoder_out=proto_x,
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )
        # print(z_ret.keys())
        # exit()
        return encoder_out, all_z_ret

    def build_vq_input(self, vq_input_x, x_embed, vq_input_type, vq, padding_mask, prev_vq_tokens=None):

        vq_input_code = None

        if self.mask_predictor_input == 'all':
            vq_input = torch.zeros_like(vq_input_x).cuda()
        else:
            if vq_input_type == "prev_target":
                vq_input = vq_input_x
            elif vq_input_type == "prev_embed":
                vq_input = x_embed
            elif vq_input_type.startswith("prev_vq_target"):
                if prev_vq_tokens is None:
                    vq_input, vq_idx, _, _ = self.quantize(
                        vq_input_x,
                        inputs_idx=None,
                        mask=~padding_mask,
                        vq=vq,
                        update_code=not self.dont_update_code_elsewhere
                    )

                else:
                    vq_input = self.forward_code(prev_vq_tokens, vq_input_code=vq_input_code, vq=vq) # vq_input_code.forward(indices=prev_vq_tokens)
            else:
                raise NotImplementedError

        return vq_input

    def quantize(self, inputs, inputs_idx, mask=None, vq=None, update_code=True):
        quantized, vq_idx, loss, dist = vq(inputs, x_idx=inputs_idx, mask=mask, update_code=update_code)
        return quantized, vq_idx, loss, dist

    def update_code(self, posterior):
        """ EMA update """
        self.code.update_code(posterior)

    def update_output_code(self, posterior):
        """ EMA update """
        # z_e, idx, mask
        self.output_code.update_code(posterior)

    def forward_code(self, idx, vq_input_code=None, vq=None, detach_code=True):
        if vq_input_code is not None:
            q = vq_input_code.forward(indices=idx)
        else:
            q = vq.forward_code(idx.unsqueeze(0), detach_code=detach_code).squeeze(0)
        return q

    def forward_z(self, predictor=None, vq=None, src_tokens=None, inputs=None, x=None, encoder_padding_mask=None):
        if src_tokens is not None:
            # vector quantization from the reference --- non-parameter posterior
            inference_out, idx = self._inference_z(vq, x, src_tokens, encoder_padding_mask)
        else:
            inference_out, idx = None, None

        if self.predictor is None:
            # a non-parameterize predictor, nearest search with decoder inputs
            predict_out = None
        else:
            # a parameterize predictor, we use GLAT here.
            if self.predict_masked_code:
                masked_positions = []
                lengths = torch.sum(~encoder_padding_mask, dim=-1).detach().cpu().numpy()-1
                for length in lengths:
                    masked_positions.append(torch.randint(low=0, high=length, size=(1,)))
                masked_positions = torch.cat(masked_positions)
                # masked_positions = torch.stack([torch.arange(unmasked_tokens.size(0)), masked_positions], dim=0)
                mask = torch.zeros_like(idx)
                mask[[torch.arange(src_tokens.size(0)), masked_positions]] = 1
                mask = mask.bool()

                predict_out, idx = self._predict_masked_z(
                    predictor,
                    unmasked_tokens=src_tokens,
                    unmasked_code_embed=inference_out["z_q_st"],
                    unmasked_codes=idx,
                    encoder_padding_mask=encoder_padding_mask,
                    mask=mask,
                    vq=vq,
                    out=inference_out
                )
                inference_out["soft_posterior"] = inference_out["soft_posterior"][mask]
            else:
                predict_out, idx = self._predict_z(
                    predictor,
                    inputs,
                    encoder_padding_mask[:, :-1],
                    tgt=idx[1:],
                    out=inference_out
                )

        if inference_out is not None:
            q = inference_out["z_q_st"]
            return q, {"prior": predict_out, "posterior": inference_out, "vq_ret": {"prior_out": idx}}
        else: ## Inference/Test time, q is the closest code embedding
            assert False, "Should never need to predict code for encoder"

    def _inference_z(self, vq, inputs, input_tokens, encoder_padding_mask):
        z_q_st, idx, vq_loss, soft_posterior = self.quantize(
            inputs,
            inputs_idx=input_tokens,
            mask=~encoder_padding_mask,
            vq=vq,
        )

        z_q = None
        soft_posterior = soft_posterior.view(inputs.size(0), inputs.size(1), soft_posterior.size(2))

        return {
                   "z_q_st": z_q_st,    # vq_st output, detached, no gradient
                   "z_q": z_q,
                   "z_e": inputs,
                   "idx": idx,
                   "mask": ~encoder_padding_mask,
                   "soft_posterior": soft_posterior,
                   "precomputed_loss": vq_loss
               }, idx

    def _predict_masked_z(self, predictor, unmasked_tokens, unmasked_code_embed, unmasked_codes, encoder_padding_mask, mask, vq, out=None):
        """ predict the latent variables """

        if self.predict_z_input == "codes":
            masked_codes = unmasked_codes.clone()
            masked_codes = masked_codes.masked_fill(mask, vq.pad_idx)
            masked_predict_input = self.forward_code(masked_codes, vq_input_code=None, vq=vq, detach_code=False)
        elif self.predict_z_input == "tokens":
            masked_tokens = unmasked_tokens.clone()
            masked_tokens = masked_tokens.masked_fill(mask, self.dictionary.unk())
            masked_predict_input, _ = self.forward_embedding(masked_tokens)
        else:
            raise NotImplementedError

        ret = predictor.forward_as_submodule(
            masked_predict_input,
            encoder_padding_mask,
            tgt_tokens=unmasked_codes
        )
        z_out, ext = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, {})
        z_idx = z_out.max(dim=-1)[1]

        if unmasked_codes is None:
            # predict latent variables while testing
            assert False
        else:
            return {
                "VQ": {
                    "out": z_out[mask],  # glancing outputs
                    "tgt": unmasked_codes[mask],  # reference target
                    "factor": self.latent_factor,
                    "mask": torch.ones_like(unmasked_codes[mask]).bool(),
                },
                "mask": mask,
           }, z_idx

    def _predict_z(self, predictor, inputs, encoder_padding_mask, tgt=None, out=None):
        """ predict the latent variables """
        ret = predictor.forward_as_submodule(
            inputs,
            encoder_padding_mask,
            encoder_out=None,
            incremental_state=None,
            tgt_tokens=tgt,
        )
        z_out, ext = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, {})
        z_idx = z_out.max(dim=-1)[1]

        mask = encoder_padding_mask

        if tgt is None:
            # predict latent variables while testing
            assert False
        else:
            z_mix = out["z_q_st"]
            return {
               "VQ": {
                   "out": z_out,  # glancing outputs
                   "tgt": tgt,  # reference target
                   "factor": self.latent_factor,
                   "mask": ~(encoder_padding_mask + ext["ref_mask"] > 0.)
               },
               "z_input": z_mix,
               "z_pred": ext.get("predict", None),
               "z_ref": out.get("z_q_st", None),
               "ref_mask": ext["ref_mask"] > 0.
           }, z_idx


    def _build_masked_predictor(self, main_args, vq):
        # main_args = self.args
        args = copy.deepcopy(main_args)
        args.encoder_layers = getattr(main_args, "enc_latent_layers", main_args.decoder_layers)
        args.encoder_embed_dim = getattr(args, "vq_predictor_embed_dim", args.decoder_embed_dim)
        args.encoder_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", args.decoder_ffn_embed_dim)
        args.encoder_attention_heads = getattr(args, "vq_predictor_attention_heads", args.decoder_attention_heads)
        args.encoder_output_dim = args.encoder_embed_dim
        args.dropout = getattr(main_args, "vq_dropout", main_args.dropout)
        args.enc_vq_use_shadow_attn = False

        embed_tokens = torch.nn.Embedding.from_pretrained(
            vq._codebook.embed[0],
            freeze=False,
            padding_idx=-1
        )

        latent_masked_predictor = TransformerEncoder4Parsing(
            args,
            dictionary=Dictionary(num_codes=args.enc_num_codes),
            embed_tokens=embed_tokens,
        )

        '''
        Had to do this weird trick to manually share the code embedding and the output_projection, 
        because code embedding has size [num_codebook, num_code_per_codebook, code_dimension].
        In this work, we only use num_codebook=1, so we share the weight of vq._codebook.embed[0] and the output_projection.weight
        '''
        shared_output_projection = myLinear(
            args.encoder_embed_dim,
            args.enc_num_codes + 1 if getattr(args, "enc_xtra_pad_code", False) else args.enc_num_codes,
            slice=0
        )
        shared_output_projection.weight = vq._codebook.embed
        latent_masked_predictor.output_projection = shared_output_projection

        if getattr(args, "share_bottom_layers", False):
            shared_layers = args.latent_layers if args.encoder_layers > args.latent_layers else args.encoder_layers
            for i in range(shared_layers):
                latent_masked_predictor.layers[i] = self.layers[i]
        return latent_masked_predictor

    def _build_predictor(self, main_args, vq):
        # main_args = self.args
        args = copy.deepcopy(main_args)
        args.share_decoder_input_output_embed = True #not getattr(main_args, "vq_split", not self.share_input_output_embed)
        args.decoder_layers = getattr(main_args, "enc_latent_layers", main_args.decoder_layers)
        args.decoder_embed_dim = getattr(args, "vq_predictor_embed_dim", args.decoder_embed_dim)
        args.decoder_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", args.decoder_ffn_embed_dim)
        args.decoder_attention_heads = getattr(args, "vq_predictor_attention_heads", args.decoder_attention_heads)
        args.decoder_output_dim = args.decoder_embed_dim
        args.encoder_embed_dim = getattr(args, "vq_struct_encoder_embed_dim", args.encoder_embed_dim)
        args.dropout = getattr(main_args, "vq_dropout", main_args.dropout)
        args.dec_vq_use_shadow_attn = False
        args.encoder_attn_k = getattr(main_args, "vq_predictor_encoder_attn_k")
        args.encoder_attn_v = getattr(main_args, "vq_predictor_encoder_attn_v")

        embed_tokens = torch.nn.Embedding.from_pretrained(
            vq._codebook.embed[0],
            freeze=False,
            padding_idx=-1
        )

        latent_decoder = TransformerDecoder4Parsing(
            args,
            dictionary=Dictionary(num_codes=args.enc_num_codes),
            embed_tokens=embed_tokens,
            no_encoder_attn=True
        )

        '''
        Had to do this weird trick to manually share the code embedding and the output_projection, 
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
        if getattr(args, "share_bottom_layers", False):
            shared_layers = args.latent_layers if args.decoder_layers > args.latent_layers else args.decoder_layers
            for i in range(shared_layers):
                latent_decoder.layers[i] = self.layers[i]
        return latent_decoder

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_proto_encoder_out = (
            encoder_out.proto_encoder_out
            if encoder_out.proto_encoder_out is None
            else encoder_out.proto_encoder_out.index_select(1, new_order)
        )
        new_vq_code = (
            encoder_out.vq_code
            if encoder_out.vq_code is None
            else encoder_out.vq_code.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            vq_code=new_vq_code,
            proto_encoder_out=new_proto_encoder_out,
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )
