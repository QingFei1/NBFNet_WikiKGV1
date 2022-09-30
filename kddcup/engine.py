import math
import logging

from tqdm import tqdm

import torch
from torch import nn
from torch import distributed as dist
from torch.utils import data as torch_data

from torchdrug import core, data, utils
from torchdrug.utils import comm
from torchdrug.core import Registry as R

logger = logging.getLogger(__name__)


@R.register("core.EngineEx")
class EngineEx(core.Engine):

    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_test_view=1, num_worker=0, distributed=True, log_interval=100):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0)

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if distributed and len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_test_view = num_test_view
        self.distributed = distributed

    def train(self, num_batch=50000, gradient_clip=None):
        # logger.warning("[%d] create sampler" % comm.get_rank())
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        # logger.warning("[%d] create dataloader" % comm.get_rank())
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        # logger.warning("[%d] create ddp model" % 1.get_rank())
        model = self.model
        if len(self.gpus) > 1:
            if not hasattr(self, "parallel_model"):
                if self.device.type == "cuda":
                    if self.distributed:
                        self.parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                                    find_unused_parameters=True)
                    else:
                        self.parallel_model = nn.parallel.DataParallel(model, device_ids=self.gpus)
                else:
                    self.parallel_model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model = self.parallel_model

        model.train()
        for epoch in self.meter(1):
            # logger.warning("[%d] set sampler epoch" % comm.get_rank())
            if sampler:
                sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(num_batch - start_id, self.gradient_interval)

            # logger.warning("[%d] start dataloader training" % comm.get_rank())
            for batch_id, batch in enumerate(dataloader):
                if batch_id == num_batch:
                    break
                # logger.warning("[%d] move batch to GPU" % comm.get_rank())
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                # logger.warning("[%d] forward model" % comm.get_rank())
                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                # logger.warning("[%d] backward model" % comm.get_rank())
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    # logger.warning("[%d] optimizer ---" % comm.get_rank())
                    if gradient_clip:
                        # logger.warning("[%d] clip gradient" % comm.get_rank())
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    # logger.warning("[%d] optimizer step" % comm.get_rank())
                    self.optimizer.step()
                    # logger.warning("[%d] optimizer zero_grad" % comm.get_rank())
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        # logger.warning("[%d] reduce" % comm.get_rank())
                        metric = comm.reduce(metric, op="mean")
                    # logger.warning("[%d] update metric" % comm.get_rank())
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(num_batch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split, log=True, keep_order=False, precise_bn=False, dump_valid=False):
        if comm.get_rank() == 0:
            if precise_bn:
                logger.warning("Computer precise BN statistics on %s" % split)
            else:
                logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        if keep_order:
            # keep order as the dataset
            sampler = None
            work_load = math.ceil(len(test_set) / comm.get_world_size())
            rank = comm.get_rank()
            work_range = range(work_load * rank, min(work_load * (rank + 1), len(test_set)))
            test_set = torch_data.Subset(test_set, work_range)
        else:
            sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        if precise_bn:
            model.train()
        else:
            model.eval()
        view_preds = []
        if dump_valid:
            f = open("false_valid.txt", "w")
        for i in range(self.num_test_view):
            preds = []
            targets = []
            for batch in tqdm(dataloader):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                pred, target = model.predict_and_target(batch)
                preds.append(pred)
                targets.append(target)

                if dump_valid:
                    pos_pred = pred.gather(-1, target.unsqueeze(-1))
                    mapping = batch["graph"].inv_mapping.flatten().cpu().numpy()
                    _pred = pred.cpu().numpy()
                    _pos_pred = pos_pred.cpu().numpy()
                    for i in range(_pred.shape[0]):
                        f.write("%d\t%d\t" % (mapping[batch["triplet"][i, 0, 0]], mapping[0]))
                        for j in range(_pred.shape[1]):
                            if _pred[i, j] > _pos_pred[i]:
                                f.write("%d\t" % mapping[j])
                        f.write("\n")

            pred = utils.cat(preds)
            target = utils.cat(targets)
            view_preds.append(pred)
        if dump_valid:
            f.close()
        # average predictions
        pred = utils.stack(view_preds, dim=0)
        pred = utils.mean(pred, dim=0)

        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log and not precise_bn:
            self.meter.log(metric)

        return metric