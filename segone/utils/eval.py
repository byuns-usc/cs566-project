from typing import Any

import torch
from torch import Tensor
from torchmetrics.functional import fbeta_score
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.segmentation import generalized_dice_score, mean_iou


class Evaluator:
    def __init__(self, preds: Tensor, targets: Tensor, num_classes: int = 2) -> None:
        """
        Initialize the Evaluator with ground truth and prediction tensors, dynamically assigning the device.
        This evaluator uses torchmetrics for metric calculations and supports both binary and multiclass data.

        Attributes:
            preds (torch.Tensor): Predicted mask with class labels (integers for multiclass or binary).
            targets (torch.Tensor): Ground truth mask with class labels (integers for multiclass or binary).
            num_classes (int): Number of classes (2 for binary classification).
        """
        assert preds.shape[0] == targets.shape[0], "Shape of ground truth and prediction must match"

        # Dynamically select device based on GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assign class attributes
        self.num_classes = num_classes
        self.task = "multiclass"
        self.input_format = "one-hot"

        # Convert one-hot encoded predictions to class labels
        self.y_pred = preds.argmax(dim=1).to(self.device)
        self.y_true = targets.to(self.device)

        print(self.y_pred.shape, self.y_true.shape)

    def mean_iou(self, **kwargs) -> Tensor:
        """
        Calculate the Mean Intersection over Union (mIOU) using torchmetrics.

        Returns:
            torch.Tensor: The mean Intersection over Union (mIOU) score.
        """
        return mean_iou(
            self.y_pred, self.y_true, num_classes=self.num_classes, input_format=self.input_format, **kwargs
        )

    def f1_score(self, beta: float = 1.0, **kwargs) -> Tensor:
        """
        Calculate the F-score using torchmetrics, the weighted harmonic mean of precision and recall.

        Args:
            beta (float): Weighting factor for precision and recall; beta < 1 prioritizes precision,
            while beta > 1 prioritizes recall.

        Returns:
            torch.Tensor: The F-score with the specified beta.
        """
        return fbeta_score(
            self.y_pred,
            self.y_true,
            task=self.task,
            num_classes=self.num_classes,
            beta=beta,
            **kwargs,
        )

    def accuracy(self, **kwargs) -> Tensor:
        """
        Calculate the accuracy using torchmetrics, the ratio of correct predictions to total predictions.

        Returns:
            torch.Tensor: The accuracy score.
        """
        return accuracy(self.y_pred, self.y_true, task=self.task, num_classes=self.num_classes, **kwargs)

    def dice_similarity(self, **kwargs) -> Tensor:
        """
        Calculate the Dice Similarity Coefficient (DSC) using torchmetrics, a measure of overlap between two samples.

        Returns:
            torch.Tensor: The Dice Similarity Coefficient.
        """
        return generalized_dice_score(
            self.y_pred, self.y_true, num_classes=self.num_classes, input_format=self.input_format, **kwargs
        )

    def evaluate_all(self) -> dict[str, Any]:
        """
        Calculate all metrics (mIOU, F-score, accuracy, and Dice Similarity) and return as a dictionary.

        Returns:
            dict[str, float]: A dictionary containing all evaluation metrics with their respective scores.
        """
        return {
            "mIoU": self.mean_iou(),
            "f1_score": self.f1_score(),
            "accuracy": self.accuracy(),
            "dice_similarity": self.dice_similarity(),
        }


# Example usage
if __name__ == "__main__":
    from torch import randint

    # Example binary ground truth and prediction tensors
    N = 5  # Number of samples
    C = 3  # Number of classes
    H, W = 128, 128  # Spatial dimensions: H*W resolution

    # Predicted one-hot encoded tensor (N, C, H, W)
    preds = randint(0, 2, (N, C, H, W))
    # Ground truth label tensor (N, H, W) with values from 0 to C-1
    target = randint(0, C, (N, H, W))

    evaluator = Evaluator(preds, target, num_classes=C)
    results = evaluator.evaluate_all()

    for metric, value in results.items():
        print(f"{metric}: {value}")
