from typing import Optional

import torch
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import base_architecture, TransformerModel, DEFAULT_MAX_TARGET_POSITIONS, DEFAULT_MAX_SOURCE_POSITIONS

from .transformer_encoder import TransformerEncoder4Parsing
from .ar_seq2seq_transformer import Transformer4Parsing
from .vq_transformer_decoder import VQTransformerDecoder4Parsing
from .vq_transformer_encoder import VQTransformerEncoder4Parsing


@register_model("vqtransformer4parsing")
class VQTransformer4Parsing(Transformer4Parsing):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.use_vq_encoder = getattr(args, "vq_encoder", False)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--vq-encoder", action="store_true")

        parser.add_argument("--vq-l2-factor", type=float, default=0.0)
        parser.add_argument("--vq-enc-l2-factor", type=float, default=0.0)
        parser.add_argument("--dec-output-vq-l1-factor", type=float, default=0.0)
        parser.add_argument("--dec-output-vq-l2-factor", type=float, default=0.0)
        parser.add_argument("--enc-output-vq-l1-factor", type=float, default=0.0)
        parser.add_argument("--enc-output-vq-l2-factor", type=float, default=0.0)
        parser.add_argument("--vq-xentropy-factor", type=float, default=0.0)
        parser.add_argument("--vq-maximize-z-entropy-factor", type=float, default=0.0)

        parser.add_argument('--vq-encoder-xentropy-factor', type=float, default=0.0)
        parser.add_argument("--vq-encoder-maximize-z-entropy-factor", type=float, default=0.0)

        parser.add_argument("--vq-predictor-embed-dim", type=int)
        parser.add_argument("--vq-predictor-ffn-embed-dim", type=int)
        parser.add_argument("--vq-predictor-attention-heads", type=int)

        parser.add_argument('--vq-enc-x-proto-x-similarity-l2-factor', type=float, default=0.0)
        parser.add_argument('--vq-dec-x-proto-x-similarity-l2-factor', type=float, default=0.0)
        parser.add_argument('--share-vq', action="store_true")


        Transformer4Parsing.add_args(parser)
        VQTransformerDecoder4Parsing.add_args(parser)
        VQTransformerEncoder4Parsing.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)

        """New Version of using ptrn decoder"""
        decoder = cls.build_decoder(
            args,
            tgt_dict,
            decoder_embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            vq=encoder.vq if getattr(args, "share_vq", False) else None,
        )

        return cls(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, no_encoder_attn, vq=None):
        return VQTransformerDecoder4Parsing(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            vq=vq
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if getattr(args, "vq_encoder", False):
            encoder = VQTransformerEncoder4Parsing(args, src_dict, embed_tokens)
        else:
            encoder = TransformerEncoder4Parsing(args, src_dict, embed_tokens)
        return encoder

    def max_positions(self):
        """Maximum length supported by the model."""
        if self.use_pretrained_encoder:
            return (self.decoder.max_positions(), self.decoder.max_positions())
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def forward_encoder(self, src_tokens, src_lengths, return_all_hiddens=False):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        return encoder_out

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            prev_vq_tokens=None,
            tgt_tokens=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.forward_encoder(src_tokens, src_lengths, return_all_hiddens)

        if self.use_vq_encoder:
            encoder_out, encoder_ext = encoder_out
        else:
            encoder_ext = None

        decoder_out = self.decoder(
            prev_output_tokens,
            prev_vq_tokens,
            tgt_tokens=tgt_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        mask = tgt_tokens.ne(self.decoder.padding_idx)
        src_mask = src_tokens.ne(self.encoder.padding_idx)

        model_ret, ext = self._compute_loss(decoder_out, tgt_tokens, mask)

        if ext is not None:
            # add latent prediction loss
            if "prior" in ext and "VQ" in ext["prior"]:
                model_ret["vq-L1"] = ext["prior"]["VQ"]

            if "output_VQ" in ext:
                model_ret["dec-out-vq-L1"] = ext["output_VQ"]
                model_ret["dec-out-vq-L1"]['factor'] = self.args.dec_output_vq_l1_factor
                if "posterior" in ext["output_VQ"] and self.args.dec_output_vq_l2_factor > 0:
                    model_ret["dec-out-vq-L2"] = self._compute_commitment_loss(
                        ext["output_VQ"]['posterior'],
                        mask,
                        self.args.dec_output_vq_l2_factor
                    )

            if "posterior" in ext and self.args.vq_l2_factor > 0:
                model_ret["vq-L2"] = self._compute_commitment_loss(ext['posterior'], mask, self.args.vq_l2_factor)

            if "posterior" in ext and self.args.vq_xentropy_factor > 0:
                xentropy_dict = self._compute_cross_entropy_loss(
                    ext['posterior']['soft_posterior'],#[:, 1:].contiguous(),
                    ext['prior']['VQ']['out'],
                    mask,#[:, 1:].contiguous(),
                    factor=self.args.vq_xentropy_factor
                )
                model_ret["kl_div"] = {
                    'loss': xentropy_dict['kl_div'],
                    'factor': self.args.vq_xentropy_factor
                }
                model_ret["posterior_entropy"] = {
                    'loss': xentropy_dict['posterior_entropy'],
                    'factor': self.args.vq_xentropy_factor
                }

            if "posterior" in ext and self.args.vq_maximize_z_entropy_factor > 0:
                model_ret["vq_z_entropy"] = self._compute_z_entropy_loss(
                    ext['posterior']['soft_posterior'],
                    mask,
                    self.args.vq_maximize_z_entropy_factor,
                )

            if self.args.vq_dec_x_proto_x_similarity_l2_factor > 0:
                for _il in range(self.args.decoder_layers):
                    if f"x_proto_x_similarity_{_il}" in ext:
                        model_ret[f"dec-x-proto-x-l2-layer{_il+1}"] = self._compute_commitment_loss(
                            ext[f'x_proto_x_similarity_{_il}'],
                            mask, # BUG: previously using src_mask, which is None here
                            self.args.vq_dec_x_proto_x_similarity_l2_factor
                        )

        if getattr(self.args, "vq_encoder", False) and encoder_ext is not None:

            if "posterior" in encoder_ext["z"] and self.args.vq_enc_l2_factor > 0:
                model_ret["vq-enc-L2"] = self._compute_commitment_loss(
                    encoder_ext["z"]['posterior'],
                    None, #src_mask,
                    self.args.vq_enc_l2_factor
                )

            if "output_VQ" in encoder_ext["z"]:
                model_ret["enc-out-vq-L1"] = encoder_ext["z"]["output_VQ"]
                model_ret["enc-out-vq-L1"]['factor'] = self.args.enc_output_vq_l1_factor
                if "posterior" in encoder_ext["z"]["output_VQ"] and self.args.enc_output_vq_l2_factor > 0:
                    model_ret["enc-out-vq-L2"] = self._compute_commitment_loss(
                        encoder_ext["z"]["output_VQ"]['posterior'],
                        None,
                        self.args.enc_output_vq_l2_factor
                    )

            if self.args.vq_encoder_xentropy_factor > 0:
                if self.encoder.predict_masked_code:
                    xentropy_dict = self._compute_cross_entropy_loss(
                        encoder_ext["z"]['posterior']['soft_posterior'],
                        encoder_ext["z"]['prior']['VQ']['out'],
                        mask=encoder_ext["z"]['prior']['VQ']['mask'],
                        factor=self.args.vq_encoder_xentropy_factor,
                        ignore_idx_in_posterior=self.encoder.vq.pad_idx if self.encoder.vq.xtra_pad_code else None
                    )
                else:
                    xentropy_dict = self._compute_cross_entropy_loss(
                        encoder_ext["z"]['posterior']['soft_posterior'][:, 1:].contiguous(),
                        encoder_ext["z"]['prior']['VQ']['out'],
                        src_mask[:, 1:].contiguous(),
                        factor=self.args.vq_encoder_xentropy_factor
                    )
                model_ret["enc_kl_div"] = {
                    'loss': xentropy_dict['kl_div'],
                    'factor': self.args.vq_encoder_xentropy_factor
                }
                model_ret["enc_posterior_entropy"] = {
                    'loss': xentropy_dict['posterior_entropy'],
                    'factor': self.args.vq_encoder_xentropy_factor
                }


            if self.args.vq_encoder_maximize_z_entropy_factor > 0:
                if self.encoder.predict_masked_code:
                    if src_tokens.size(1) > 2:
                        model_ret["vq_enc_z_entropy"] = self._compute_z_entropy_loss(
                            encoder_ext["z"]['posterior']['soft_posterior'], #[:, 1:].contiguous(),
                            encoder_ext["z"]['prior']['VQ']['mask'],
                            self.args.vq_encoder_maximize_z_entropy_factor,
                        )
                else:
                    model_ret["vq_enc_z_entropy"] = self._compute_z_entropy_loss(
                        encoder_ext["z"]['posterior']['soft_posterior'], #[:, 1:].contiguous(),
                        src_mask, #[:, 1:].contiguous(),
                        self.args.vq_encoder_maximize_z_entropy_factor,
                    )

            if self.args.vq_enc_x_proto_x_similarity_l2_factor > 0:
                for _il in range(self.args.encoder_layers):
                    if f'x_proto_x_similarity_{_il}' in encoder_ext:
                        model_ret[f"enc-x-proto-x-l2-layer{_il+1}"] = self._compute_commitment_loss(
                            encoder_ext[f'x_proto_x_similarity_{_il}'],
                            src_mask,
                            self.args.vq_enc_x_proto_x_similarity_l2_factor
                        )

        return model_ret

    def _compute_z_entropy_loss(self, posterior, mask, factor):
        def _compute_qz(posterior, mask):
            # if not no_sum_over_batch:
            posterior_dist = F.softmax(posterior, dim=-1).view(-1, posterior.size(-1))[mask.view(-1)]
            qz = torch.sum(posterior_dist, dim=0) / posterior_dist.size(0)  #
            # else:
            #     posterior_dist = F.softmax(posterior, dim=-1) * mask.unsqueeze(-1).float()
            #     qz = torch.sum(posterior_dist, dim=1) / posterior_dist.size(1)

            return qz

        qz = _compute_qz(posterior, mask)
        qz += 1e-30  # Prevent NaN

        z_entropy = self.entropy(qz, is_logits=False)

        # if no_sum_over_batch:
        #     z_entropy = torch.mean(z_entropy)

        loss = - z_entropy

        return {
            'loss': loss * factor,
            'factor': factor,
        }

    def _compute_cross_entropy_loss(self, posterior, prior, mask, factor, ignore_idx_in_posterior=None):
        nclass = posterior.size(-1)
        kldiv = self._compute_kldiv(posterior.view(-1, nclass), prior.view(-1, nclass), mask)
        q_entropy = self.entropy(posterior, is_logits=True, mask=mask, ignore_idx_in_posterior=ignore_idx_in_posterior)
        cross_entropy = kldiv + q_entropy
        return {
            'loss': cross_entropy.sum() / cross_entropy.size(0) * factor,
            'kl_div': kldiv.sum() / kldiv.size(0) * factor,
            'posterior_entropy': q_entropy.sum() / q_entropy.size(0) * factor,
            'factor': factor,
        }

    def entropy(self, input, is_logits=True, mask=None, ignore_idx_in_posterior=None):
        if ignore_idx_in_posterior:
            assert input.dim() == 2
            input = input[:, :ignore_idx_in_posterior]

        if is_logits:
            entropy = - torch.sum(F.log_softmax(input, dim=-1) * F.softmax(input, dim=-1), dim=-1)
        else:
            entropy = - torch.sum(torch.log(input) * input, dim=-1)

        if mask is not None:
            entropy = entropy[mask]
        else:
            entropy = entropy.view(-1)
        return entropy

    def _compute_orthogonal_reg_loss(self, posterior):
        return {
            'loss': posterior['precomputed_loss']['orthogonal_reg_loss'],
            'factor': 1.0
        }

    def _compute_kldiv(self, posterior, prior, mask, logits=True, reduction=True):
        if logits:
            input = F.log_softmax(prior, dim=-1)
            target = F.softmax(posterior, dim=-1)
            # log_target = F.log_softmax(posterior, dim=-1)
        else:
            input = torch.log(prior)
            target = posterior
            # log_target = torch.log(posterior)

        # if self.split_kldiv_gradient:
        #     kl_div = 1/2 * F.kl_div(input.detach(), target, log_target=False, reduction='none') \
        #         + 1/2 * F.kl_div(input, target.detach(), log_target=False, reduction='none')
        # else:
        kl_div = F.kl_div(input, target, log_target=False, reduction='none')

        if reduction:
            kl_div = kl_div.sum(dim=-1).view(-1)[mask.view(-1)]
            return kl_div
        else:
            return kl_div


    def _compute_kldiv_loss(self, posterior, prior, mask, factor, aggregate=True, logits=True):
        kl_div = self._compute_kldiv(posterior, prior, mask, logits=logits)
        if aggregate:
            kl_div_agg = kl_div.sum() / kl_div.size(0)
        else:
            kl_div_agg = kl_div
        return {
            'loss': kl_div_agg * factor,
            'factor': factor,
        }

    def _compute_commitment_loss(self, posterior, mask, factor):
        commit_loss = F.mse_loss(posterior['z_q_st'].detach(), posterior['z_e'], reduction = 'none')
        commit_loss = commit_loss[mask].mean()
        return {
            'loss': commit_loss * factor,
            'factor': factor
        }

    def _compute_loss(self, decode_out, tgt_tokens, mask):
        if isinstance(decode_out, tuple):
            decode_out, inner_out = decode_out[0], decode_out[1]
        else:
            inner_out = None

        if inner_out is not None and "ref_mask" in inner_out:
            word_ins_mask = (inner_out["ref_mask"].squeeze(-1) < 1.0) * mask  # non reference and non padding
        else:
            word_ins_mask = mask  # non padding

        model_ret = {
            "word_ins": {"out": decode_out, "tgt": tgt_tokens, "mask": word_ins_mask, "ls": self.args.label_smoothing,
                         "nll_loss": True}
        }

        return model_ret, inner_out


@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnStructEnc_2l2h32d64ffnVQ")
def vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnStructEnc_2l2h32d64ffnVQ(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.latent_layers = getattr(args, "latent_layers", 2)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 2)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 32)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 64)

    args.vq_struct_encoder_layers = getattr(args, "vq_struct_encoder_layers", 2)
    args.vq_struct_encoder_attention_heads = getattr(args, "vq_struct_encoder_attention_heads", 2)
    args.vq_struct_encoder_embed_dim = getattr(args, "vq_struct_encoder_embed_dim", 32)
    args.vq_struct_encoder_ffn_embed_dim = getattr(args, "vq_struct_encoder_ffn_embed_dim", 64)

    return base_architecture(args)


@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_3l4h256d512ffnStructEnc_2l4h256d512ffnVQ")
def vq_transformer4parsing_3l4h256d512ffn_3l4h256d512ffnStructEnc_2l4h256d512ffnVQ(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.latent_layers = getattr(args, "latent_layers", 2)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 4)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 256)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 512)

    args.vq_struct_encoder_layers = getattr(args, "vq_struct_encoder_layers", 3)
    args.vq_struct_encoder_attention_heads = getattr(args, "vq_struct_encoder_attention_heads", 4)
    args.vq_struct_encoder_embed_dim = getattr(args, "vq_struct_encoder_embed_dim", 256)
    args.vq_struct_encoder_ffn_embed_dim = getattr(args, "vq_struct_encoder_ffn_embed_dim", 512)

    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnStructEnc_2l4h256d512ffnVQ")
def vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnStructEnc_2l4h256d512ffnVQ(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.latent_layers = getattr(args, "latent_layers", 2)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 4)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 256)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 512)

    args.vq_struct_encoder_layers = getattr(args, "vq_struct_encoder_layers", 2)
    args.vq_struct_encoder_attention_heads = getattr(args, "vq_struct_encoder_attention_heads", 2)
    args.vq_struct_encoder_embed_dim = getattr(args, "vq_struct_encoder_embed_dim", 32)
    args.vq_struct_encoder_ffn_embed_dim = getattr(args, "vq_struct_encoder_ffn_embed_dim", 64)

    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnVQ")
def vq_transformer4parsing_3l4h256d512ffn_2l2h32d64ffnVQ(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.latent_layers = getattr(args, "latent_layers", 2)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 2)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 32)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 64)

    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_4l8h256d1024ffnVQ")
def vq_transformer4parsing_3l4h256d512ffn_4l8h256d1024ffnVQ(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.latent_layers = getattr(args, "latent_layers", 4)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 8)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 256)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 1024)

    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_2l2h64d")
def vq_transformer_2l2h64d(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.latent_layers = getattr(args, "latent_layers", 2)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_2l2h32d64ffn")
def vq_transformer4parsing_2l2h32d64ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.latent_layers = getattr(args, "latent_layers", 2)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_2l2h64d_6lvq")
def vq_transformer_2l2h64d_6lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.latent_layers = getattr(args, "latent_layers", 6)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_2l2h64d512ffn")
def vq_transformer_2l2h64d512ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.latent_layers = getattr(args, "latent_layers", 2)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn")
def vq_transformer_3l4h256d512ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.latent_layers = getattr(args, "latent_layers", 2)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_3lvq")
def vq_transformer_3l4h256d512ffn_3lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.latent_layers = getattr(args, "latent_layers", 3)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_4lvq")
def vq_transformer_3l4h256d512ffn_4lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.latent_layers = getattr(args, "latent_layers", 4)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_5lvq")
def vq_transformer_3l4h256d512ffn_5lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.latent_layers = getattr(args, "latent_layers", 5)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_3l4h256d512ffn_6lvq")
def vq_transformer_3l4h256d512ffn_6lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.latent_layers = getattr(args, "latent_layers", 6)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_4lvq")
def vq_transformer_6l8h512d1024ffn_6lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.latent_layers = getattr(args, "latent_layers", 4)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_6lvq")
def vq_transformer_6l8h512d1024ffn_6lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.latent_layers = getattr(args, "latent_layers", 6)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_4l8h256d1024ffnvq")
def vq_transformer4parsing_6l8h512d1024ffn_4l8h256d1024ffnvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)

    args.latent_layers = getattr(args, "latent_layers", 4)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 8)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 256)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 1024)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_2l4h256d512ffnvq")
def vq_transformer4parsing_6l8h512d1024ffn_2l4h256d512ffnvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)

    args.latent_layers = getattr(args, "latent_layers", 2)
    args.vq_predictor_attention_heads = getattr(args, "vq_predictor_attention_heads", 4)
    args.vq_predictor_embed_dim = getattr(args, "vq_predictor_embed_dim", 256)
    args.vq_predictor_ffn_embed_dim = getattr(args, "vq_predictor_ffn_embed_dim", 512)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_12lvq")
def vq_transformer_6l8h512d1024ffn_12lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.latent_layers = getattr(args, "latent_layers", 12)
    return base_architecture(args)

@register_model_architecture("vqtransformer4parsing", "vq_transformer4parsing_6l8h512d1024ffn_2lvq")
def vq_transformer_6l8h512d1024ffn_2lvq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.latent_layers = getattr(args, "latent_layers", 2)
    return base_architecture(args)