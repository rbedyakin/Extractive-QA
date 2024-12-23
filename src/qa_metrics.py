from torchmetrics import Metric
import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional, Union, Callable
from torchmetrics.utilities.compute import _safe_divide
from collections import Counter


def precision_recall_f1(
        pred_sum: torch.Tensor, tp_sum: torch.Tensor, true_sum: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision, recall, f1.

        Args:
            pred_sum: Tensor of predicted positives per type.
            tp_sum: Tensor of true positives per type.
            true_sum: Tensor of actual positives per type.
        
        Returns:
            precision: Tensor of precision per type.
            recall: Tensor of recall per type. 
            f1: Tensor of f1 per type.
        
        Note: Metric value is substituted as 0 when encountering zero division."""

    precision = _safe_divide(num=tp_sum, denom=pred_sum, zero_division=0.0)
    recall = _safe_divide(num=tp_sum, denom=true_sum, zero_division=0.0)
    f1 = _safe_divide(num=2 * tp_sum,
                      denom=pred_sum + true_sum,
                      zero_division=0.0)

    return precision, recall, f1


class QA_Metric(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self, stage: Optional[str] = None, **kwargs):
        """Init Metric

        Args:
            stage: Optional prefix for keys in output dict
                default: None
        """
        super().__init__(**kwargs)

        self.stage = stage

        self.add_state("correct",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("pred_sum",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("tp_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_sum",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("sum_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: List[str], target: List[str]) -> None:
        """Update state with predictions and targets.
        
        Args:
            preds: List of predictions (Estimated targets as returned by a tagger)
            target: List of reference (Ground truth (correct) target values)
        """
        for pred, gt in zip(preds, target):
            self.correct += torch.tensor(int(pred == gt))
            self.total += torch.tensor(1)

            gt_tokens = gt.split()
            pred_tokens = pred.split()
            common_tokens = Counter(gt_tokens) & Counter(pred_tokens)
            pred_sum = torch.tensor(len(pred_tokens))
            tp_sum = torch.tensor(sum(common_tokens.values()))
            true_sum = torch.tensor(len(gt_tokens))
            _, _, f1 = precision_recall_f1(pred_sum=pred_sum,
                                           tp_sum=tp_sum,
                                           true_sum=true_sum)
            self.sum_f1 += f1
            self.pred_sum += pred_sum
            self.tp_sum += tp_sum
            self.true_sum += true_sum

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the final statistics.
        
        Returns:
            'metrics': dict. Summary of the scores."""

        metrics = {}
        metrics[f'{self.stage}_exact_match'] = _safe_divide(num=self.correct,
                                                            denom=self.total,
                                                            zero_division=0.0)
        metrics[f'{self.stage}_macro_f1'] = _safe_divide(num=self.sum_f1,
                                                         denom=self.total,
                                                         zero_division=0.0)
        precision, recall, f1 = precision_recall_f1(pred_sum=self.pred_sum,
                                                    tp_sum=self.tp_sum,
                                                    true_sum=self.true_sum)
        metrics[f'{self.stage}_overall_precision'] = precision
        metrics[f'{self.stage}_overall_recall'] = recall
        metrics[f'{self.stage}_overall_f1'] = f1

        return metrics
