import torch
from torch import Tensor
from torchmetrics.functional import fbeta_score
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.segmentation import mean_iou, generalized_dice_score


class Evaluator:
    def __init__(self, y_true: Tensor, y_pred: Tensor, num_classes: int = 2, task="multiclass") -> None:
        """
        Initialize the Evaluator with ground truth and prediction tensors, dynamically assigning the device.
        This evaluator uses torchmetrics for metric calculations and supports both binary and multiclass data.

        Args:
            y_true (torch.Tensor): Ground truth mask with class labels (integers for multiclass or binary).
            y_pred (torch.Tensor): Predicted mask with class labels (integers for multiclass or binary).
            num_classes (int): Number of classes (2 for binary classification).
        """
        assert y_true.shape == y_pred.shape, "Shape of ground truth and prediction must match"

        # Dynamically select device based on GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.task = task

        self.y_true = y_true.to(self.device)
        self.y_pred = y_pred.to(self.device)

    def mIOU(self, **kwargs) -> Tensor:
        """
        Calculate the Mean Intersection over Union (mIOU) using torchmetrics.

        Returns:
            torch.Tensor: The mean Intersection over Union (mIOU) score.
        """
        return mean_iou(self.y_pred, self.y_true, num_classes=self.num_classes, **kwargs)

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
        return generalized_dice_score(self.y_pred, self.y_true, num_classes=self.num_classes, **kwargs)

    def evaluate_all(self) -> dict[str, float]:
        """
        Calculate all metrics (mIOU, F-score, accuracy, and Dice Similarity) and return as a dictionary.

        Returns:
            dict[str, float]: A dictionary containing all evaluation metrics with their respective scores.
        """
        return {
            "mIOU": self.mIOU(),
            "f1_score": self.f1_score(),
            "accuracy": self.accuracy(),
            "dice_similarity": self.dice_similarity(),
        }


# Example usage
if __name__ == "__main__":
    from torch import randint

    # Example binary ground truth and prediction tensors
    N = 5  # Number of samples
    classes = 3  # Number of classes

    preds = randint(-1, 1, (N, classes, 16, 16))  # 10 samples, 3 classes, 16x16 resolution
    target = randint(-1, 1, (N, classes, 16, 16))

    evaluator = Evaluator(preds, target, num_classes=3)
    results = evaluator.evaluate_all()

    for metric, value in results.items():
        print(f"{metric}: {value}")
