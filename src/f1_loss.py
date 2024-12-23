import torch
import torch.nn.functional as F


def f1_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    '''Calculate soft multiclass F1 loss.
    
    The original implementation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 
        0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    '''
    # Check for invalid values in y_pred and replace them with a safe value
    # y_pred = torch.nan_to_num(y_pred, torch.tensor(0.0))

    y_true = F.one_hot(y_true, num_classes=y_pred.shape[-1])
    y_pred = F.softmax(y_pred, dim=-1)

    # [..., 1:] - takes all batches and all other dims, but labels without zero index,

    tp = (y_true[..., 1:] * y_pred[..., 1:]).sum()
    fp = ((1 - y_true[..., 1:]) * y_pred[..., 1:]).sum()
    fn = (y_true[..., 1:] * (1 - y_pred[..., 1:])).sum()

    # print(f'{tp=}, {fp=}, {fn=}')

    epsilon = torch.tensor(1e-7)

    # f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = 2 * tp / (2 * tp + fp + fn + epsilon)
    return 1 - f1


if __name__ == "__main__":
    batch_size = 1
    step_size = 2
    num_labels = 3
    y_true = torch.randint(low=0,
                           high=num_labels,
                           size=(batch_size, step_size))
    y_pred = torch.randn((batch_size, step_size, num_labels))

    print(f'{y_true=}')
    print(f'{y_pred=}')

    loss = f1_loss(y_pred=y_pred, y_true=y_true)
    print(f'{loss=}')
