import os
from typing import Optional, Dict

import torch

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX.summary import hparams
from tensorboardX import SummaryWriter
from d3rlpy.algos.base import AlgoBase
from d3rlpy.logger import D3RLPyLogger, LOG


class ImprovedAlgoBase(AlgoBase):

    def _prepare_logger(
        self,
        save_metrics: bool,
        experiment_name: Optional[str],
        with_timestamp: bool,
        logdir: str,
        verbose: bool,
        tensorboard_dir: Optional[str],
    ) -> D3RLPyLogger:
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = ImprovedD3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        return logger


class ImprovedD3RLPyLogger(D3RLPyLogger):
    def __init__(
        self,
        experiment_name: str,
        tensorboard_dir: Optional[str] = None,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        super().__init__(experiment_name, None, save_metrics, root_dir, verbose, with_timestamp)

        if tensorboard_dir:
            tfboard_path = os.path.join(
                tensorboard_dir, self._experiment_name
            )
            self._writer = CorrectedSummaryWriter(logdir=tfboard_path)
        else:
            self._writer = None

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print("%d,%d,%f" % (epoch, step, metric), file=f)

                if self._writer:
                    self._writer.add_scalar(f"metrics/{name}", metric, epoch)

            metrics[name] = metric

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        # if self._params and self._writer:
        #     self._writer.add_hparams(
        #         self._params,
        #         metrics,
        #         name=self._experiment_name,
        #         global_step=epoch,
        #     )

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)