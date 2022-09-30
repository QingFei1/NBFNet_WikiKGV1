import os
import sys
import time
import logging

import yaml
import easydict
import jinja2

import torch
from torch import distributed as dist

from torchdrug.utils import comm, sparse_coo_tensor

def meshgrid(dict):
    if len(dict) == 0:
        yield {}
        return

    key = next(iter(dict))
    values = dict[key]
    sub_dict = dict.copy()
    sub_dict.pop(key)

    if not isinstance(values, list):
        values = [values]
    for value in values:
        for result in meshgrid(sub_dict):
            result[key] = value
            yield result


def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()

    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        for hyperparam in meshgrid(grid):
            config = easydict.EasyDict(yaml.load(template.render(hyperparam)))
            configs.append(config)
    else:
        configs = [easydict.EasyDict(yaml.load(raw_text))]

    return configs


def sparse_tensor_index_select(tensor, index, dim=0):
    _indices = tensor._indices()
    _values = tensor._values()
    _shape = list(tensor.shape)

    index_set, inverse = torch.unique(index, return_inverse=True)
    index2id = -torch.ones(_shape[dim], dtype=torch.long, device=tensor.device)
    index2id[index] = torch.arange(len(index), device=tensor.device)

    valid = index2id[_indices[dim]] >= 0
    _indices = _indices[:, valid]
    _indices[dim] = index2id[_indices[dim]]
    _values = _values[valid]
    _shape[dim] = len(index)
    return sparse_coo_tensor(_indices, _values, _shape), index2id


def sparse_tensor_flatten(tensor, start=0, end=1):
    assert end == start + 1
    _indices = tensor._indices()
    _values = tensor._values()
    _shape = list(tensor.shape)
    
    _indices[end] = _indices[start] * _shape[end] + _indices[end]
    _indices = torch.cat([_indices[:start], _indices[end:]])

    _shape[end] = _shape[start] * _shape[end]
    _shape = _shape[:start] + _shape[end:]
    return sparse_coo_tensor(_indices, _values, _shape)


class _ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if comm.get_rank() > 0:
            while True:
                pass

        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode="Plain", color_scheme="Linux", call_pdb=1)
        return self.instance(*args, **kwargs)


def setup_hook():
    sys.excepthook = _ExceptionHook()


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    # if console:
    #     handler = logging.StreamHandler(sys.stdout)
    #     handler.setFormatter(format)
    #     logger.addHandler(handler)
    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg, name=None):
    name = name or ""
    file_name = "%s_working_dir" % os.environ["SLURM_JOB_ID"]
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(cfg.output_dir, cfg.task["class"], cfg.task.model["class"],
                              name + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    if world_size > 1:
        dist.barrier()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    if world_size > 1:
        dist.barrier()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(output_dir)
    return output_dir