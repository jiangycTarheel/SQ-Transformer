import math

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


class Accuracy(object):
    def __init__(self, counts=0, totals=0):
        self.counts = counts
        self.totals = totals

    @property
    def score(self):
        if self.totals <= 0:
            return 0.0
        else:
            return 100.0 * self.counts / self.totals

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in [self.counts, self.totals]])
        return 'Accuracy = {score:.{width}f} {precisions}'.format(
            score=self.score,
            width=width,
            precisions=precisions,
        )

    def __iter__(self):
        return self

    def __add__(self, other):
        if isinstance(other, int):
            return self
        return Accuracy(
            counts=self.counts + other.counts,
            totals=self.totals + other.totals,
        )

    def __radd__(self, other):
        return self + other

    @classmethod
    def compute_acc(cls, tgt, out=None, mask=None, pred=None):
        # assert out is not None and tgt is not None, "output and target should not be empty"
        if out is None or tgt is None:
            return Accuracy()
        tgt = tgt.view(-1)
        if pred is None and out is not None:
            out = out.view(-1, out.size(-1))
            pred = out.max(-1)[1]

        indicate = pred.eq(tgt)

        if mask is not None:
            mask = mask.view(-1)
            indicate = indicate * mask
            totals = mask.sum().item()
        else:
            totals = tgt.size(0)
        counts = indicate.sum().item()

        return Accuracy(counts=counts, totals=totals)


@register_criterion("generic_vq_cross_entropy")
class GenericVQXECriterion(LabelSmoothedCrossEntropyCriterion):
    def _compute_acc(self, acc, name="acc"):
        return {"name": name, "acc": acc}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens, src_lengths, prev_output_tokens = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            sample["net_input"]["prev_output_tokens"]
        )

        tgt_tokens = sample["target"]

        net_output = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens=tgt_tokens)
        losses, nll_loss = [], []
        accuracies = []

        for obj in net_output:
            if net_output[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    net_output[obj].get("out"),
                    net_output[obj].get("tgt"),
                    net_output[obj].get("mask", None),
                    net_output[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=net_output[obj].get("factor", 1.0)
                )
            else:
                _losses = self._custom_loss(
                    net_output[obj].get("loss"),
                    name=obj + '-loss',
                    factor=net_output[obj].get("factor", 1.0)
                )
                # print(obj)
                # print(net_output[obj].get("factor", 1.0))
            if not model.args.no_accuracy and not net_output[obj].get("no-acc", False):
                if not model.training and "tgt" in net_output[obj]:  # compute only for evaluation
                    acc = self._compute_acc(
                        acc=Accuracy.compute_acc(
                            pred=net_output[obj].get("pred", None),
                            out=net_output[obj].get("out", None),
                            tgt=net_output[obj].get("tgt", None),
                            mask=net_output[obj].get("mask", None),
                        ),
                        name=obj + '-acc'
                    )
                    accuracies += [acc]

            losses += [_losses]
            if net_output[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
        for l in accuracies:
            logging_output[l["name"]] = l["acc"]

        return loss, sample_size, logging_output

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            elif "acc" in key:
                val = sum([log.get(key, Accuracy()) for log in logging_outputs], Accuracy())
                metrics.log_scalar(key, val.score, nsentences, round=2)


@register_criterion("generic_cross_entropy")
class GenericXECriterion(CrossEntropyCriterion):
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        n_sent = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        # real_sample_size = utils.item(sum(log.get("real_sample_size", 0) for log in logging_outputs))

        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            elif "acc" in key:
                val = sum([log.get(key, Accuracy()) for log in logging_outputs], Accuracy())
                metrics.log_scalar(key, val.score, n_sent, round=2)


@register_criterion("generic_label_smoothed_cross_entropy")
class GenericLabelSmoothedXECriterion(LabelSmoothedCrossEntropyCriterion):
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        n_sent = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            elif "acc" in key:
                val = sum([log.get(key, Accuracy()) for log in logging_outputs], Accuracy())
                metrics.log_scalar(key, val.score, n_sent, round=2)