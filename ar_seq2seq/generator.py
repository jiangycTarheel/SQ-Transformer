from typing import Dict, List, Optional
import math

import torch
from torch import Tensor

from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.models.fairseq_encoder import EncoderOut

from .vq_transformer_encoder import VQTransformerEncoder4Parsing
from .vq_transformer_decoder import VQTransformerDecoder4Parsing

class VQSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        infer_with_tgt=False,
    ):
        if isinstance(models, EnsembleVAEModel):
            models = models
        else:
            models = EnsembleVAEModel(models)

        super().__init__(models, tgt_dict, beam_size, max_len_a, max_len_b, min_len, normalize_scores, len_penalty,
            unk_penalty, temperature, match_source_len, no_repeat_ngram_size, search_strategy, eos,
            symbols_to_strip_from_output, lm_model, lm_weight,
        )
        self.infer_with_tgt = infer_with_tgt

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
        debug=False
    ):

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)

        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        if isinstance(self.model.models[0].decoder, VQTransformerDecoder4Parsing):
            codes = [(
                torch.zeros(bsz * beam_size * 1, max_len + 2)
                    .to(src_tokens)
                    .long()
                    .fill_(-1)
            )]  # +2 for eos and pad
            codes[0][:, 0] = self.eos if bos_token is None else bos_token
            # posterior_codes = [(
            #     torch.zeros(bsz, max_len + 2)
            #         .to(src_tokens)
            #         .long()
            #         .fill_(-1)
            # )]  # +2 for eos and pad
            # posterior_codes[0][:, 0] = self.eos if bos_token is None else bos_token
        else:
            raise NotImplementedError

        # posterior_codes[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size * 1).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)

        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        tgt_tokens = sample["target"] if "target" in sample else None #and self.infer_with_tgt else None
        if tgt_tokens is not None:
            tgt_tokens_repeat = tgt_tokens.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size * 1, -1)
            # posterior_codes, tgt_lengths = self.model.infer_posterior_code(tgt_tokens)#.transpose(0, 1)

        # else:
        #     posterior_codes, tgt_lengths = None, None

        has_mismatch_code = False

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if debug:
                print(f'step: {step}')

            if reorder_state is not None:
                # print(reorder_state[:10])
                # print('\n')
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )  # e.g., [0, 1, 2, 4, 5] - [0, 1, 2, 3, 4] = [0, 0, 0, 1, 1]

                    reorder_state.view(-1, beam_size * 1).add_(
                        corr.unsqueeze(-1) * beam_size * 1 * 1
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]

                # dec_reorder_state = reorder_state.unsqueeze(1).repeat(1, code_cand_size).reshape(-1)
                dec_reorder_state = reorder_state
                # self.model.reorder_incremental_state(incremental_states, reorder_state // code_cand_size, dec_reorder_state)
                self.model.reorder_incremental_state(incremental_states, reorder_state, dec_reorder_state)


                # encoder_outs_for_vq = self.model.reorder_encoder_out(
                #     encoder_outs_for_vq, reorder_state // code_beam_size
                # )
                # encoder_outs_for_dec = self.model.reorder_encoder_out(
                #     encoder_outs_for_dec, dec_reorder_state
                # )
                encoder_outs_for_vq, encoder_outs_for_dec = [None], [None]
                # new_order = torch.arange(tokens.size(0)).unsqueeze(1).repeat(1, code_beam_size).view(
                #     -1).contiguous().long().cuda()

            if step == 0:
                encoder_outs_for_vq = encoder_outs
                new_order = torch.arange(tokens.size(0)).contiguous().long().cuda()
                encoder_outs_for_dec = self.model.reorder_encoder_out(encoder_outs, new_order)
                # print(encoder_outs_for_vq[0].encoder_out.size())
                # print(encoder_outs_for_dec[0].encoder_out.size())
                # print('\n')

            lprobs, avg_attn_scores, code_idx_dict, has_mismatch_code = self.model.forward_decoder(
                step,
                tokens[:, : step + 1],
                # codes,
                # code_cand_size=code_cand_size,
                encoder_outs_for_vq=encoder_outs_for_vq,
                encoder_outs_for_dec=encoder_outs_for_dec,
                incremental_states=incremental_states,
                temperature=self.temperature,
                tgt_tokens=tgt_tokens_repeat[:, : step + 1] if tgt_tokens is not None else None,
                infer_with_tgt=self.infer_with_tgt,
                # posterior_codes=posterior_codes[0],
                has_mismatch_code=has_mismatch_code,
            )
            # code_idx = None
            # if code_idx_dict['code_idx'] is not None:
            #     code_idx = [_code_idx['prior'].view(bsz, beam_size*code_beam_size, code_cand_size) \
            #                 for _code_idx in code_idx_dict['code_idx']]
                # posterior_code_idx = code_idx_dict['code_idx']['posterior'].view(bsz, beam_size*code_beam_size)
                # if posterior_code_idx is not None:
                #     posterior_codes[:, step + 1] = posterior_code_idx[:, 0]

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            # print(lprobs.view(bsz, -1, self.vocab_size).size())
            # print(scores.view(bsz, beam_size * code_beam_size, -1)[:, :, :step].size())
            # exit()
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                # beam_size,
                # code_beam_size,
                # code_cand_size,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]

            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # if code_idx is not None:
            #     print(code_idx)
            #     print(cand_bbsz_idx)
            #     code_idx_selected = torch.index_select(
            #         code_idx.view(-1), dim=0, index=cand_bbsz_idx.view(-1)
            #     )
            #     codes[:, : step + 1] = torch.index_select(
            #         codes[:, : step + 1], dim=0, index=cand_bbsz_idx.view(-1) // code_beam_size  # [0, 1, 2] in the expande
            #     )
            #     codes.view(bsz, beam_size * code_beam_size, -1)[:, :, step + 1] = code_idx_selected.view(bsz, beam_size * code_beam_size)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size * code_beam_size * 2)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            # print(cand_scores.size())
            # print(cand_indices.size())
            # print(cand_beams.size())
            # print(eos_mask.size())
            # print(cands_to_ignore.size())
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    codes,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    # posterior_codes,
                    # tgt_lengths
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                codes = [_codes.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1) for _codes in codes]
                if tgt_tokens is not None:
                    tgt_tokens_repeat = tgt_tokens_repeat.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            # print(eos_mask.type_as(cand_offsets) * cand_size)
            # print(cand_offsets[: eos_mask.size(1)])
            # exit()
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            # print(active_bbsz_idx)
            # print(codes[:code_beam_size])
            # print('\n')
            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            # if code_idx is not None:
            #     code_idx_selected = [torch.index_select(
            #         _code_idx.view(-1), dim=0, index=active_bbsz_idx
            #     ) for _code_idx in code_idx]
            #
            #     for _codes, _code_idx_selected in zip(codes, code_idx_selected):
            #         _codes[:, : step + 1] = torch.index_select(
            #             _codes[:, : step + 1], dim=0, index=active_bbsz_idx // code_cand_size  # [0, 1, 2] in the expande
            #         )
            #         _codes.view(bsz, beam_size * code_beam_size, -1)[:, :, step + 1] = _code_idx_selected.view(bsz, beam_size * code_beam_size)

            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx  # [0, 1, 2] in the expande
            )
            #index = active_bbsz_idx // code_beam_size since [0, 1, 2] in the expanded idx are actually all generated from the tokens[0, :]
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx


            # print(posterior_codes)
            # exit()
            # posterior_codes[0][(posterior_codes[0]==-1).nonzero()[0]] = -1

            # if not torch.all(torch.eq(codes[0], posterior_codes[0])):
            #     print(posterior_codes)
            #     print(codes)
            #     print('\n')


        # sort by score descending
        for sent in range(len(finalized)):
            # scores = torch.tensor(
            #     [float(elem["score"].item()) for elem in finalized[sent]]
            # )
            # _, sorted_scores_indices_before = torch.sort(scores, descending=True)
            # if debug:
            #     posterior_code = posterior_codes[sent][:tgt_lengths[sent]]  # [:-1]
            #     print('posterior_code:')
            #     print(posterior_code)
            #     changed_score = False
            #     for ie, elem in enumerate(finalized[sent]):
            #         pred_codes = elem['codes']
            #         print(pred_codes)
            #         print(elem['score'])
            #         # exit()
            #         # if pred_codes.size(0) == posterior_code.size(0) and torch.all(torch.eq(pred_codes, posterior_code)):
            #         #     elem['score'] += torch.tensor(10).to(elem['score'])

            # print('\n')
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            # if not torch.all(torch.eq(sorted_scores_indices, sorted_scores_indices_before)):
            #     print(scores)
            #     print(sorted_scores_indices_before)
            #     print(sorted_scores_indices)
            # if sorted_scores_indices[0] != sorted_scores_indices_before[0]:
            #     print(scores)
            #     print(sorted_scores_indices_before)
            #     print(sorted_scores_indices)

            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )

        return finalized

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        codes,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
        # posterior_codes,
        # tgt_lengths,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS
        codes_clone = [_codes.index_select(0, bbsz_idx)[
            :, 1: step + 1
        ] for _codes in codes]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        sents_seen: Dict[str, Optional[Tensor]] = {}
        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # print(f"{step} FINISHED {idx} {score} {sent}={unfin_idx} {cum_unfin}")
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                # print('posterior_codes', len(posterior_codes))
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "codes": [_codes_clone[i] for _codes_clone in codes_clone],
                        # "posterior_codes": [_pos_codes[sent][:tgt_lengths[sent]][:-1] for _pos_codes in posterior_codes],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished


class MySequenceGenerator(SequenceGenerator):

    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        if isinstance(models, EnsembleBaselineModel):
            models = models
        else:
            models = EnsembleBaselineModel(models)

        super().__init__(models, tgt_dict, beam_size, max_len_a, max_len_b, min_len, normalize_scores, len_penalty,
            unk_penalty, temperature, match_source_len, no_repeat_ngram_size, search_strategy, eos,
            symbols_to_strip_from_output, lm_model, lm_weight,
        )

class EnsembleBaselineModel(EnsembleModel):

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                self.reorder_pretrained_encoder_out(encoder_outs[i], new_order) if model.use_pretrained_encoder is not None \
                    else model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    def reorder_pretrained_encoder_out(self, encoder_out: EncoderOut, new_order):
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
        # encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        # new_encoder_embedding = (
        #     encoder_embedding
        #     if encoder_embedding is None
        #     else encoder_embedding.index_select(0, new_order)
        # )
        # src_tokens = encoder_out.src_tokens
        # if src_tokens is not None:
        #     src_tokens = src_tokens.index_select(0, new_order)
        #
        # src_lengths = encoder_out.src_lengths
        # if src_lengths is not None:
        #     src_lengths = src_lengths.index_select(0, new_order)
        #
        # encoder_states = encoder_out.encoder_states
        # if encoder_states is not None:
        #     for idx, state in enumerate(encoder_states):
        #         encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,  # B x T
            src_lengths=None,  # B x 1
        )

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        # print(net_input['src_tokens'][0])
        # jump_embed = self.models[0].encoder.embed_tokens.weight[16]
        # jump_embed = self.models[0].encoder.embed_tokens.weight[76]
        # jump_embed = self.models[0].encoder.embed_tokens.weight[616]
        # walk_embed = self.models[0].encoder.embed_tokens.weight[13]
        # look_embed = self.models[0].encoder.embed_tokens.weight[14]
        # run_embed = self.models[0].encoder.embed_tokens.weight[15]
        # print(torch.nn.functional.cosine_similarity(jump_embed, walk_embed, dim=0))
        # print(torch.nn.functional.cosine_similarity(jump_embed, run_embed, dim=0))
        # print(torch.nn.functional.cosine_similarity(jump_embed, look_embed, dim=0))
        # exit()
        # q = 6
        # U, S, V = torch.pca_lowrank(self.models[0].encoder.embed_tokens.weight, q=q)
        #
        # print([torch.dot(jump_embed, V[:,_q])/torch.norm(V[:,_q])
        #        for _q in range(q)])
        # print([torch.dot(walk_embed, V[:, _q]) / torch.norm(V[:, _q]) for _q in range(q)])
        # print([torch.dot(look_embed, V[:, _q]) / torch.norm(V[:, _q]) for _q in range(q)])
        # print([torch.dot(run_embed, V[:, _q]) / torch.norm(V[:, _q]) for _q in range(q)])
        # print(torch.dot(jump_embed, V[:, 2]) / torch.norm(V[:, 2]))
        # print(torch.nn.functional.cosine_similarity(jump_embed, V[:,0], dim=0))
        # print(torch.nn.functional.cosine_similarity(jump_embed, V[:, 1], dim=0))
        # print(torch.nn.functional.cosine_similarity(jump_embed, V[:, 2], dim=0))
        # exit()
        if not self.has_encoder():
            return None
        if self.models[0].use_pretrained_encoder:
            return [model.forward_encoder(net_input['src_tokens'], net_input['src_lengths']) for model in self.models]
        elif isinstance(self.models[0].encoder, VQTransformerEncoder4Parsing):
            return [model.encoder.forward_torchscript(net_input)[0] for model in self.models]
        else:
            return [model.encoder.forward_torchscript(net_input) for model in self.models]


class EnsembleVAEModel(EnsembleModel):

    # def infer_posterior_code(self, tgt_tokens):
    #     if tgt_tokens is not None and self.models[0].tgt_encoder is not None:
    #         tgt_lengths = torch.sum(tgt_tokens.ne(self.models[0].tgt_encoder.padding_idx), dim=1, dtype=torch.int)
    #     else:
    #         tgt_lengths = torch.sum(tgt_tokens.ne(self.models[0].decoder.padding_idx), dim=1, dtype=torch.int)
    #
    #     if tgt_tokens is not None and self.models[0].tgt_encoder is not None:
    #         tgt_encoder_out = self.models[0].tgt_encoder(
    #             tgt_tokens, tgt_lengths
    #         )
    #     else:
    #         tgt_encoder_out = None
    #
    #     if self.models_size == 1:
    #         posterior_code = self.models[0].decoder.infer_posterior_code(tgt_tokens, tgt_encoder_out) #.transpose(0, 1)
    #         return posterior_code, tgt_lengths
    #     else:
    #         raise NotImplementedError
    #
    # def forward_tgt_encoder(self, tgt_tokens):
    #     tgt_lengths = torch.sum(tgt_tokens.ne(self.tgt_encoder.padding_idx), dim=1, dtype=torch.int)
    #     # tgt_encoder_out = self.tgt_encoder(
    #     #     tgt_tokens, tgt_lengths
    #     # )
    #     return [model.tgt_encoder(tgt_tokens, tgt_lengths) for model in self.models]

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        if isinstance(self.models[0].encoder, VQTransformerEncoder4Parsing):
            return [model.encoder.forward_torchscript(net_input)[0] for model in self.models]
        else:
            return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        step,
        tokens,
        encoder_outs_for_vq: List[EncoderOut],
        encoder_outs_for_dec: List[EncoderOut],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        tgt_tokens=None,
        overwrite_self_attn_state=True,
        overwrite_cross_attn_state=True,
        infer_with_tgt=False,
        has_mismatch_code=False,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out_for_vq: Optional[EncoderOut] = None
        encoder_out_for_dec: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out_for_vq = encoder_outs_for_vq[i]
                encoder_out_for_dec = encoder_outs_for_dec[i]

            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.decode(
                    tokens,
                    # code_beam_size=code_cand_size,
                    encoder_out_for_vq=encoder_out_for_vq,
                    encoder_out_for_dec=encoder_out_for_dec,
                    incremental_state=incremental_states[i],
                    tgt_tokens=tgt_tokens,
                    overwrite_self_attn_state=overwrite_self_attn_state,
                    overwrite_cross_attn_state=overwrite_cross_attn_state,
                    infer_with_tgt=infer_with_tgt
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            code_idx = None
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                    # code_idx = decoder_out[1]["vq"]["idx"]['prior']
                    # posterior_code_idx = decoder_out[1]["vq"]["idx"]['posterior']
                    code_lprobs = decoder_out[1]["vq"][-1]["lprobs"] if 'vq' in decoder_out[1] else None
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            # print(probs)
            # print(probs.size())
            # if code_lprobs is not None:
            #     probs += code_lprobs.view(-1).unsqueeze(1).unsqueeze(2) * 2000
                # print(code_lprobs)
                # code_probs = model.get_normalized_probs(
                #     (code_scores, None), log_probs=False, sample=None
                # )
                # print(code_probs)
                # print(code_probs.size())
                # print('\n')

            # print(probs[:, -1, :].size())
            # print(probs.size())
            # exit()
            probs = probs[:, -1, :]

            # if debug:
            #     if not has_mismatch_code:
            #         code_beam_time_beam_size = 6
            #         bsz = tgt_tokens.size(0) // code_beam_time_beam_size
                    # code_beam_size = 3
                    # bsz = tgt_tokens.size(0) // code_beam_size

                    # for _ib in range(bsz):
                    #     if decoder_out[1]["vq"][0]["idx"]['prior'].view(bsz, code_beam_time_beam_size, code_cand_size)[_ib, 0, 0] != \
                    #             decoder_out[1]["vq"][0]["idx"]['posterior'].view(bsz, code_beam_time_beam_size)[_ib, 0]:
                    #         has_mismatch_code = True

                    # if has_mismatch_code:
                    #     print(step)
                    #     # print('code beams:')
                    #     # for codes in code_beams[0]:
                    #     #     print(codes)
                    #     print('posterior codes:')
                    #     print(posterior_codes)
                    #     print('predicted codes:')
                    #     print(decoder_out[1]["vq"][0]["idx"]['prior'].view(bsz, code_beam_time_beam_size, code_cand_size))
                    #     print(code_lprobs.view(bsz, code_beam_time_beam_size, code_cand_size))
                    #     print('\n')

            # print(decoder_out[1]["vq"]["idx"]['prior'].view(4, 6, code_cand_size))
            # print(decoder_out[1]["vq"]["idx"]['posterior'])
            # exit()
            if self.models_size == 1:
                if "vq" in decoder_out[1]:
                    return probs, attn, {
                        'code_idx': [decoder_out[1]["vq"][_id]["idx"] for _id in range(len(decoder_out[1]["vq"]))]}, has_mismatch_code
                else:
                    return probs, attn, {'code_idx': None}, has_mismatch_code

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                self.reorder_pretrained_encoder_out(encoder_outs[i], new_order) if model.use_pretrained_encoder is not None \
                else model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_struct_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                self.reorder_pretrained_encoder_out(encoder_outs[i], new_order) if model.use_pretrained_struct_encoder is not None \
                    else model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    def reorder_pretrained_encoder_out(self, encoder_out: EncoderOut, new_order):
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
        # encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        # new_encoder_embedding = (
        #     encoder_embedding
        #     if encoder_embedding is None
        #     else encoder_embedding.index_select(0, new_order)
        # )
        # src_tokens = encoder_out.src_tokens
        # if src_tokens is not None:
        #     src_tokens = src_tokens.index_select(0, new_order)
        #
        # src_lengths = encoder_out.src_lengths
        # if src_lengths is not None:
        #     src_lengths = src_lengths.index_select(0, new_order)
        #
        # encoder_states = encoder_out.encoder_states
        # if encoder_states is not None:
        #     for idx, state in enumerate(encoder_states):
        #         encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,  # B x T
            src_lengths=None,  # B x 1
        )

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            vq_new_order,
            new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order, vq_new_order
            )