from typing import Optional
import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import base_architecture, TransformerModel
from fairseq.models.nat.nonautoregressive_transformer import utils

from .transformer_decoder import TransformerDecoder4Parsing
from .transformer_encoder import TransformerEncoder4Parsing


@register_model("transformer4parsing")
class Transformer4Parsing(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.use_pretrained_encoder = getattr(args, "use_pretrained_encoder", None)
        self.dont_update_encoder = getattr(args, "dont_update_encoder", False)
        if self.dont_update_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--mask-decoder-input', type=str, default=None,
                            help='path to pre-trained decoder embedding')
        parser.add_argument("--block-cls", type=str, default="None")
        parser.add_argument("--self-attn-cls", type=str, default="abs")
        parser.add_argument("--enc-block-cls", type=str, default="abs")
        parser.add_argument("--enc-self-attn-cls", type=str, default="abs")
        parser.add_argument("--max-rel-positions", type=int, default=20)
        # parser.add_argument("--ptrn-model-path", type=str, default=None)
        # parser.add_argument("--use-ptrn-encoder", action="store_true")
        # parser.add_argument("--use-ptrn-decoder", action="store_true")
        # parser.add_argument('--use-pretrained-encoder', type=str, default=None)
        # parser.add_argument("--dont-update-encoder", action="store_true")
        # parser.add_argument("--enc-sparsity-factor", type=float, default=0)
        # parser.add_argument("--dec-sparsity-factor", type=float, default=0)
        TransformerModel.add_args(parser)
        TransformerDecoder4Parsing.add_args(parser)
        TransformerEncoder4Parsing.add_args(parser)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
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
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if getattr(args, "use_pretrained_encoder", None) is not None:
            if args.use_pretrained_encoder == 't5-v1_1-large':
                args.encoder_embed_dim = 1024
            elif args.use_pretrained_encoder in ['t5-v1_1-xl', 't5-v1_1-xxl']:
                args.encoder_embed_dim = 2048

        return TransformerDecoder4Parsing(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder4Parsing(args, src_dict, embed_tokens)
        return encoder

    def forward_encoder(self, src_tokens, src_lengths, return_all_hiddens=False):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        return encoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        if self.use_pretrained_encoder:
            return (self.decoder.max_positions(), self.decoder.max_positions())
        return (self.encoder.max_positions(), self.decoder.max_positions())

    @classmethod
    def load_pretrained_model(cls, args):
        if getattr(args, "ptrn_model_path", None) is None:
            return None

        from fairseq import checkpoint_utils
        models, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.ptrn_model_path),
            task=None,
            suffix=getattr(args, "checkpoint_suffix", ""),
        )
        return models[0]

@register_model_architecture("transformer4parsing", "transformer4parsing_2l2h64d")
def transformer_2l2h64d(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    return base_architecture(args)

@register_model_architecture("transformer4parsing", "transformer4parsing_2l2h64d512ffn")
def transformer_2l2h64d512ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    return base_architecture(args)

@register_model_architecture("transformer4parsing", "transformer4parsing_3l4h256d512ffn")
def transformer_3l4h256d512ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    return base_architecture(args)

@register_model_architecture("transformer4parsing", "transformer4parsing_6l8h512d1024ffn")
def transformer_6l8h512d1024ffn(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    return base_architecture(args)