import json
import logging
from argparse import Namespace
import os

import torch
from fairseq import utils
from fairseq.data import (
    encoders,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, FairseqDataset
from fairseq.data import iterators
from fairseq import metrics, search, tokenizer

from .vq_seq2seq_transformer import VQTransformer4Parsing
from fairseq.search import BeamSearch

EVAL_BLEU_ORDER = 4
EVAL_ROUGE_ORDER = ['1', '2', 'l']

logger = logging.getLogger(__name__)

class Accuracy(object):
    def __init__(self, counts, totals):
        self.counts = counts
        self.totals = totals

class Rouge_obj(object):
    def __init__(self, sys_len, ref_len, scores, total):
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.scores = scores
        self.total = total

@register_task('seq2seq-translation')
class Seq2SeqTranslationTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument("--data-name", type=str, required=True)
        parser.add_argument("--no-accuracy", action='store_true', default=False)
        parser.add_argument("--eval-overall-acc", action="store_true", default=False)
        parser.add_argument("--eval-acc-args", type=str)
        parser.add_argument("--extract-code", action='store_true', default=False)
        parser.add_argument('--infer-with-tgt', default=False, action='store_true')
        parser.add_argument("--eval-cognition", action='store_true', default=False)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            gen_args = Namespace(**gen_args)
            # gen_args.iter_decode_eos_penalty = getattr(args, 'iter_decode_eos_penalty', 0.0)
            # gen_args.iter_decode_max_iter = getattr(args, 'iter_decode_max_iter', 10)
            # gen_args.iter_decode_beam = getattr(args, 'iter_decode_with_beam', 1)
            # gen_args.iter_decode_external_reranker = getattr(args, 'iter_decode_with_external_reranker', False)
            # gen_args.decoding_format = getattr(args, 'decoding_format', None)
            # gen_args.iter_decode_force_max_iter = getattr(args, 'iter_decode_force_max_iter', False)
            # gen_args.retain_history = getattr(args, 'retain_iter_history', False)
            gen_args.infer_with_tgt = getattr(args, "infer_with_tgt", False)
            self.sequence_generator = self.build_generator([model], gen_args)
        elif getattr(args, 'eval_overall_acc', False):
            # self.tokenizer = encoders.build_tokenizer(Namespace(
            #     tokenizer=getattr(args, 'eval_bleu_detok', None),
            #     **detok_args
            # ))
            #
            gen_args = json.loads(getattr(args, 'eval_acc_args', '{}') or '{}')
            gen_args = Namespace(**gen_args)

            # gen_args.iter_decode_eos_penalty = getattr(args, 'iter_decode_eos_penalty', 0.0)
            # gen_args.iter_decode_max_iter = getattr(args, 'iter_decode_max_iter', 10)
            # gen_args.iter_decode_beam = getattr(args, 'iter_decode_with_beam', 1)
            # gen_args.iter_decode_external_reranker = getattr(args, 'iter_decode_with_external_reranker', False)
            # gen_args.decoding_format = getattr(args, 'decoding_format', None)
            # gen_args.iter_decode_force_max_iter = getattr(args, 'iter_decode_force_max_iter', False)
            # gen_args.retain_history = getattr(args, 'retain_iter_history', False)
            gen_args.infer_with_tgt = getattr(args, "infer_with_tgt", False)
            self.sequence_generator = self.build_generator([model], gen_args)

        return model


    def _inference_with_bleu(self, generator, sample, model, compute_code_acc=False):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            if getattr(self.args, 'eval_cognition', False):
                s = " ".join(list(s.replace(" ", "")))
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def _inference_with_acc(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            # if self.tokenizer:
            #     s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        acc_count = 0

        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))

            if hyps[-1] == refs[-1]:
                acc_count += 1

        return Accuracy(acc_count, len(hyps))
    #

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            model.eval()
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model, compute_code_acc=True)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
            model.train()
        elif self.args.eval_overall_acc:
            model.eval()
            acc_obj = self._inference_with_acc(self.sequence_generator, sample, model)
            logging_output['overall_acc'] = acc_obj

            #logging_output['overall_acc_totals'] = acc_totals
            # logging_output['_acc_sys_len'] = acc.sys_len
            # logging_output['_acc_ref_len'] = acc.ref_len
            model.train()

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        from fairseq.sequence_generator import (
            # SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        from .generator import VQSequenceGenerator, MySequenceGenerator

        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if isinstance(models[0], VQTransformer4Parsing):
            search_strategy = BeamSearch(self.target_dictionary)
            seq_gen_cls = VQSequenceGenerator

            extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

            return seq_gen_cls(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                search_strategy=search_strategy,
                infer_with_tgt=getattr(args, "infer_with_tgt", False),
                # code_beam_size=getattr(args, "code_beam_size", 1),
                # code_cand_size=getattr(args, "code_cand_size", 1),
                **extra_gen_cls_kwargs,
            )
        else:
            if sampling:
                search_strategy = search.Sampling(
                    self.target_dictionary, sampling_topk, sampling_topp
                )
            elif diverse_beam_groups > 0:
                search_strategy = search.DiverseBeamSearch(
                    self.target_dictionary, diverse_beam_groups, diverse_beam_strength
                )
            elif match_source_len:
                # this is useful for tagging applications where the output
                # length should match the input length, so we hardcode the
                # length constraints for simplicity
                search_strategy = search.LengthConstrainedBeamSearch(
                    self.target_dictionary,
                    min_len_a=1,
                    min_len_b=0,
                    max_len_a=1,
                    max_len_b=0,
                )
            elif diversity_rate > -1:
                search_strategy = search.DiverseSiblingsSearch(
                    self.target_dictionary, diversity_rate
                )
            elif constrained:
                search_strategy = search.LexicallyConstrainedBeamSearch(
                    self.target_dictionary, args.constraints
                )
            elif prefix_allowed_tokens_fn:
                search_strategy = search.PrefixConstrainedBeamSearch(
                    self.target_dictionary, prefix_allowed_tokens_fn
                )
            else:
                search_strategy = search.BeamSearch(self.target_dictionary)

            if seq_gen_cls is None:
                if getattr(args, "at_generator", False):
                    seq_gen_cls = VQSequenceGenerator
                elif getattr(args, "print_alignment", False):
                    seq_gen_cls = SequenceGeneratorWithAlignment
                    # print('alignment')
                else:
                    # print('seqgenerator')
                    seq_gen_cls = MySequenceGenerator

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

