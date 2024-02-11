import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, NamedTuple
import torch.nn.functional as F

from fairseq.models.transformer import TransformerEncoder
from nat.layer import BlockedEncoderLayer
from nat.modules import RelativePositionEmbeddings

def build_relative_embeddings(args, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = args.decoder_embed_dim // getattr(args, "decoder_attention_heads")
    return RelativePositionEmbeddings(
        max_rel_positions=getattr(args, "max_rel_positions", 4),
        embedding_dim=embedding_dim,
        direction=True,
        dropout=args.dropout
    )

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("attention_weights", Optional[Tensor])
    ],
)

class TransformerEncoder4Parsing(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "enc_self_attn_cls", "abs") != "abs":
            self.embed_positions = None
            rel_keys = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            rel_vals = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            self.layers = torch.nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_encoder_layer(args, _il, rel_keys, rel_vals)
                    for _il in range(args.encoder_layers)
                ]
            )

    def build_encoder_layer(self, args, ilayer=-1, rel_keys=None, rel_vals=None):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return BlockedEncoderLayer(args)
        else:
            return BlockedEncoderLayer(
                args,
                relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
            )

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        return_head_attentions: bool = False,
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

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        attention_weights = []
        for layer in self.layers:
            x, layer_dict = layer(x, encoder_padding_mask, need_head_weights=return_head_attentions)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            if return_head_attentions:
                attention_weights.append(layer_dict['attn'])

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
            attention_weights=attention_weights
        )

    def forward_as_submodule(
        self,
        x_input,
        self_attn_padding_mask,
        return_all_hiddens: bool = False,
        normalize: bool = False,
        tgt_tokens=None,
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

        # B x T x C -> T x B x C
        x = x_input.transpose(0, 1)

        # compute padding mask
        # encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        features = x.transpose(0, 1)
        ret = {}
        ret["ref_mask"] = self_attn_padding_mask
        ret["inputs"] = x_input
        ret["features"] = features
        encoder_out = self.output_layer(features)

        ret["predict"] = encoder_out.max(dim=-1)

        encoder_out = F.log_softmax(encoder_out, -1) if normalize else encoder_out

        if tgt_tokens is not None:
            return encoder_out, ret
        else:
            return encoder_out


    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)


    @staticmethod
    def add_args(parser):
        parser.add_argument("--enc-no-bias-for-self-attn", action="store_true")
