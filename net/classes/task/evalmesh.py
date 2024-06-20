import json
from ssl import RAND_pseudo_bytes
from typing import Dict, Tuple, List
from runner import Runner
from task.task import Task
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from checkpoint_io import CheckpointIO
import bisect
from helper import get_class_from_string, slice_dict
import time

class EvalMesh(Task):

    def __init__(self, config):
        self.last_time :float = 0
        self.epochs : int = 1000
        self.learning_rate :float= 1e-4
        self.linear_growth : bool = True
        self.multi = False
        self.update_checkpoint :int = 100
        self.pin_memory : bool = True
        self.clip :float = None
        self.resume_from : str = 'Train_latest.ckpt'
        self.resume_epoch : int = None
        self.optimizer : str = "Adam"
        self.phase : Dict[str, List[Tuple[float, float]]] = {}  # {network: [[phase, lr_factor]...]]
        self.batch_size : int = 1
        self.force_run : bool = False
        self.old_train : bool = False
        super().__init__(config)




    def __call__(self):
        super().__call__()

        if not hasattr(self, 'data') and hasattr(self.runner, 'data'):
            self.data = self.runner.data

        if not hasattr(self, 'loss') and hasattr(self.runner, 'loss'):
            self.loss = self.runner.loss

        self.runner.py_logger.info(json.dumps(Runner.filterdict(self.runner.key_value.copy()), indent=4, skipkeys=True, sort_keys=True))
        self.runner.logger.log_config()

        self.runner.py_logger.info(f"Start training for {self.epochs} epochs")
        # copy entire code to logs
        self.runner.logger.log_code_base()

        self.runner.network.cuda()

        Optim = get_class_from_string("torch.optim."+self.optimizer)
        self._param_groups = list(self.phase.keys()) if len(self.phase) > 0 else [name for name, _ in self.runner.network.named_children()]
        if self.old_train:
            optimizer = torch.optim.Adam(lr=1e-4,params=self.runner.network.parameters())
        else:
            optimizer = Optim(lr=self.learning_rate, params=[{'params': [p for p in getattr(self.runner.network, name).parameters()]} for name in self._param_groups], betas=(0.1, 0.9))

        self.ckpt_io = CheckpointIO(self.folder.replace("EvalMesh","Train"), network=self.runner.network, optimizer=optimizer)

        len(self.data)
        dataset = DataLoader(self.data, batch_size=self.batch_size,
                             pin_memory=self.pin_memory, num_workers=1 if self.multi else 0)

        # resume
        try:
            load_dict = self.ckpt_io.load(self.resume_from)
        except Exception as e:
            self.runner.py_logger.warn(repr(e))
            load_dict = dict()

        ep = load_dict.get('epoch', 0)

        if self.resume_epoch is not None:
            ep = self.resume_epoch

        ep -= 1

        input_points_t = torch.Tensor([[-0.0744043,-0.137061,0.282421]])
        query_feature_coords, relative_coords, query_has_feature = self.runner.data.getFeatureCoords(input_points_t.unsqueeze(0))


        model_input = next(iter(dataset))
        self.runner.evaluator.epoch_hook(ep,slice_dict(model_input, [0]))
        try:
            model_input = next(iter(dataset))
            self.runner.evaluator.epoch_hook(ep,slice_dict(model_input, [0]))
        except Exception as e:
            self.runner.py_logger.warn(repr(e))

        try:
            self.runner.evaluator.epoch_hook(ep)
        except Exception as e:
            self.runner.py_logger.warn(repr(e))



