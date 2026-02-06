from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial
import numpy as np
import pandas as pd
import pingouin
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy.typing as npt

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, ensure_tuple, Average, look_up_option
from monai.metrics import (
    CumulativeIterationMetric,
    MAEMetric,
    ROCAUCMetric,
)
from monai.metrics.confusion_matrix import (
    compute_confusion_matrix_metric,
    get_confusion_matrix,
    ConfusionMatrixMetric,
)
from monai.networks.utils import one_hot


class ICCMetric(CumulativeIterationMetric):
    """ """

    def __init__(self, average: Average | str = Average.MACRO) -> None:
        super().__init__()
        self.average = average

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return y_pred, y

    def aggregate(
        self, average: Average | str | None = None
    ) -> np.ndarray | float | npt.ArrayLike:
        """
        Typically `y_pred` and `y` are stored in the cumulative buffers at each iteration,
        This function reads the buffers and computes the area under the ROC.

        Args:
            average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
                Type of averaging performed if not binary classification. Defaults to `self.average`.

        """
        y_pred, y = self.get_buffer()
        # compute final value and do metric reduction
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")

        y_pred = y_pred.flatten()
        y = y.flatten()
        n_samples = len(y)
        if n_samples < 5:
            f = torch.tensor(0.0)
        else:
            icc_df = pd.DataFrame(
                data={
                    "targets": [*np.arange(n_samples)] * 2,
                    "ratings": y.cpu().numpy().tolist() + y_pred.cpu().numpy().tolist(),
                    "raters": [1] * n_samples + [2] * n_samples,
                }
            )

            icc_stat = pingouin.intraclass_corr(
                data=icc_df, targets="targets", raters="raters", ratings="ratings"
            )
            f = icc_stat[icc_stat["Type"] == "ICC2"]["ICC"].item()
            f = torch.tensor(f)
        return f


class PearsonCorMetric(CumulativeIterationMetric):
    """ """

    def __init__(self, average: Average | str = Average.MACRO) -> None:
        super().__init__()
        self.average = average

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return y_pred, y

    def aggregate(
        self, average: Average | str | None = None
    ) -> np.ndarray | float | npt.ArrayLike:
        """
        Typically `y_pred` and `y` are stored in the cumulative buffers at each iteration,
        This function reads the buffers and computes the area under the ROC.

        Args:
            average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
                Type of averaging performed if not binary classification. Defaults to `self.average`.

        """
        y_pred, y = self.get_buffer()
        # compute final value and do metric reduction
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")

        f = torch.corrcoef(torch.stack([y.flatten(), y_pred.flatten()]))[0, 1]
        return f


class MyRegressionMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        value_for_nan: float = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.value_for_nan = value_for_nan

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.shape != y.shape:
            raise ValueError(
                f"y_pred and y shapes dont match, received y_pred: [{y_pred.shape}] and y: [{y.shape}]"
            )

        # also check if there is atleast one non-batch dimension i.e. num_dims >= 2
        if len(y_pred.shape) < 2:
            raise ValueError(
                "either channel or spatial dimensions required, found only batch dimension"
            )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        self._check_shape(y_pred, y)
        return self._compute_metric(y_pred, y)

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        nd = len(data)
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        if not_nans < nd:
            f = torch.tensor(
                self.value_for_nan or torch.nan, device=f.device
            ).broadcast_to(f.shape)
        return (f, not_nans) if self.get_not_nans else f


class NegativeRMSEMetric(MyRegressionMetric):
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        flt = partial(torch.flatten, start_dim=1)
        f = (y - y_pred) ** 2
        f = -torch.sqrt(torch.mean(flt(f), dim=-1, keepdim=True))
        return f


class NegativeMAEMetric(MyRegressionMetric):
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        flt = partial(torch.flatten, start_dim=1)
        f = torch.abs(y - y_pred)
        f = -torch.mean(flt(f), dim=-1, keepdim=True)
        return f
    
class NegativeMAPEMetric(MyRegressionMetric):
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        flt = partial(torch.flatten, start_dim=1)
        f = torch.abs(y - y_pred)
        f = f / (y + 1e-5)
        f = -torch.mean(flt(f), dim=-1, keepdim=True)
        return f


class NegativeMSEMetric(MyRegressionMetric):
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        flt = partial(torch.flatten, start_dim=1)
        f = (y - y_pred) ** 2
        f = -torch.mean(flt(f), dim=-1, keepdim=True)
        return f


class ProbConfusionMatrixMetric(ConfusionMatrixMetric):
    """
    nonbinarized predictions. otherwise, same as monai.metrics.ConfusionMatrixMetrics
    """

    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "hit_rate",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn(
                    "As for classification task, compute_sample should be False."
                )
                self.compute_sample = False

        y_pred_bin = one_hot(y_pred.argmax(1, keepdim=True), y_pred.shape[1])

        return get_confusion_matrix(
            y_pred=y_pred_bin, y=y, include_background=self.include_background
        )
