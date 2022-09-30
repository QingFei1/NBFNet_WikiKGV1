import argparse
import getpass
import logging
import math
import os
import sys
import copy
import pprint
import shutil

import numpy as np
import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils, datasets, layers
from torchdrug.utils import comm

import util
from kg import framework, task, model, logic, dataset
from kddcup import engine, optim
from extension import sparse


layers.MessagePassingBase.gradient_checkpoint = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/oneway.yaml")
    parser.add_argument("-s", "--start", help="start config id for hyperparmeter search", type=int,
                        default=None)
    parser.add_argument("-e", "--end", help="end config id for hyperparmeter search", type=int,
                        default=None)

    return parser.parse_known_args()[0]


util.setup_hook()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfgs = util.load_config(args.config)
    cfgs[0].output_dir = cfgs[0].output_dir.replace("zhuzhaoc", getpass.getuser())

    seed = cfgs[0].get("seed", 1024)
    torch.manual_seed(seed)
    np_seed = cfgs[0].get("numpy_seed", seed)
    np.random.seed(np_seed)

    name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = util.create_working_directory(cfgs[0], name=name)
    if comm.get_rank() == 0:
        logger = util.get_root_logger()

    args.start = args.start or 0
    args.end = args.end or len(cfgs)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning("Hyperparameter grid size: %d" % len(cfgs))
        logger.warning("Current job search range: [%d, %d)" % (args.start, args.end))
        # copy the config to current working directory
        shutil.copyfile(args.config, os.path.basename(args.config))

    cfgs = cfgs[args.start: args.end]
    for job_id, cfg in enumerate(cfgs):
        # set working directory for each job in hyperparamter search
        cwd = output_dir
        if len(cfgs) > 1:
            cwd = os.path.join(cwd, str(job_id))
        if comm.get_rank() == 0:
            logger.warning("<<<<<<<<<< Job %d / %d start <<<<<<<<<<" % (job_id, len(cfgs)))
            logger.warning(pprint.pformat(cfg))
            os.makedirs(cwd, exist_ok=True)
        if comm.get_world_size() > 1:
            dist.barrier()
        os.chdir(cwd)

        if job_id == 0:
            logger.warning("[%d] start loading dataset" % comm.get_rank())
            _dataset = core.Configurable.load_config_dict(cfg.dataset)
            logger.warning("[%d] end loading dataset" % comm.get_rank())
            train_set, valid_set, test_set = _dataset.split()
            logger.warning("[%d] end split dataset" % comm.get_rank())
            full_valid_set = valid_set
            full_test_set = test_set
            valid_set = torch_data.random_split(valid_set, (len(valid_set) // 1000, len(valid_set) - len(valid_set) // 1000))[0]
            # if comm.get_rank() == 0:
            #     with open("/home/b/bengioy/zhuzhaoc/scratch/small_valid_indices.pkl", "wb") as fout:
            #         import pickle
            #         pickle.dump(valid_set.indices, fout)
            #     raise ValueError
            # test_set = torch_data.random_split(test_set, (len(test_set) // 10, len(test_set) - len(test_set) // 10))[0]
        if comm.get_rank() == 0:
            logger.warning(_dataset)
            logger.warning("using 0.1% of valid")
            logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

        if "score_model" in cfg.task.model:
            cfg.task.model.score_model.num_entity = _dataset.num_entity
            cfg.task.model.score_model.num_relation = _dataset.num_relation
            if "likelihood_model" in cfg.task.model:
                cfg.task.model.likelihood_model.num_entity = _dataset.num_entity
                cfg.task.model.likelihood_model.num_relation = _dataset.num_relation
            if "RGCN" in cfg.task.model.gnn_model["class"] or "LogicalGCN" in cfg.task.model.gnn_model["class"] or \
                                "SemiringGCN" in cfg.task.model.gnn_model["class"]:
                num_relation = _dataset.num_relation
                if cfg.task.model["class"] in \
                        ["NodeEncoder", "OnewayPropagation", "OnewayBidirectional", "SoftNeuralLP", "BellmanFord",
                        "BellmanFordKDDCup", "BatchedBellmanFordKDDCup"] \
                        and cfg.task.model.flip_edge:
                    num_relation = num_relation * 2
                cfg.task.model.gnn_model.num_relation = num_relation
            if "LinearScore" in cfg.task.model.score_model["class"]:
                cfg.task.model.score_model.output_dim = _dataset.num_relation
        else:
            cfg.task.model.num_entity = _dataset.num_entity
            cfg.task.model.num_relation = _dataset.num_relation

        _task = core.Configurable.load_config_dict(cfg.task)
        cfg.optimizer.params = _task.parameters()
        cfg.optimizer.lr = float(cfg.optimizer.lr)
        if "weight_decay" in cfg.optimizer:
            cfg.optimizer.weight_decay = float(cfg.optimizer.weight_decay)
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        if "scheduler" in cfg:
            cfg.scheduler.optimizer = optimizer
            scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        else:
            scheduler = None

        # down sample test set
        if "test" in cfg:
            if cfg.test.get("use_valid_as_test", False):
                if comm.get_rank() == 0:
                    logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    logger.warning("!!!!!!!!!!!!!! use valid as test !!!!!!!!!!!!!!")
                    logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if "range" in cfg.test:
                    test_set = copy.copy(full_valid_set)
                    test_set.t_correct_index = None
                else:
                    test_set = copy.copy(valid_set)
                    cfg.test.range = [0, len(valid_set)]
                    test_set.dataset.t_correct_index = None
            else:
                test_set = full_test_set
            test_range = range(*cfg.test.range)
            test_set = torch_data.Subset(test_set, test_range)

        solver = engine.EngineEx(_task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)
        sub_train_set = torch_data.random_split(train_set, [len(valid_set), len(train_set) - len(valid_set)])[0]

        if "checkpoint" in cfg:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.checkpoint)
            state = torch.load(cfg.checkpoint, map_location=solver.device)
            # state["model"].pop("train_graph")
            solver.model.load_state_dict(state["model"], strict=False)

            if cfg.get("tune_indicator_only", False):
                for param in solver.model.parameters():
                    param.requires_grad_(False)
                assert isinstance(solver.model.model, framework.BatchedBellmanFordKDDCup)
                solver.model.model.score_model.relation.weight.requires_grad_(True)

            # solver.optimizer.load_state_dict(state["optimizer"])
            # for state in solver.optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.to(solver.device)

            comm.synchronize()

            if cfg.train.num_batch == 0:
                batch_size = solver.batch_size
                solver.batch_size = 1

                if cfg.get("precise_bn", False):
                    solver.evaluate("valid", precise_bn=True)
                if "test" in cfg:
                    # dump submission file
                    solver.evaluate("test", log=False, keep_order=True)
                else:
                    # normal evaluation
                    solver.evaluate("valid")

                solver.batch_size = batch_size

        if cfg.train.num_batch > 0:
            # train
            num_epoch = cfg.get("num_epoch", 10)
            step = math.ceil(cfg.train.num_batch / num_epoch)
            best_score = float("-inf")
            best_epoch = -1

            for i in range(num_epoch):
                if "InductiveKnowledgeGraphEmbedding" in cfg.task["class"]:
                    _task.mode = "train"
                solver.train(num_batch=step)
                solver.save("model_epoch_%d.pth" % solver.epoch)

                if "BellmanFord" in cfg.task.model["class"]:
                    num_negative = _task.num_negative
                    _task.num_negative = 4096

                # solver.train_set = sub_train_set
                # solver.evaluate("train")
                # solver.train_set = train_set

                if "InductiveKnowledgeGraphEmbedding" in cfg.task["class"]:
                    _task.mode = "valid"
                batch_size = solver.batch_size
                solver.batch_size = 1
                metric = solver.evaluate("valid")
                solver.batch_size = batch_size
                score = metric["mrr (top-10)"]
                if score > best_score:
                    best_score = score
                    best_epoch = solver.epoch

                if "BellmanFord" in cfg.task.model["class"]:
                    _task.num_negative = num_negative
                if i < num_epoch - 1 and cfg.get("shuffle_dataset", False):
                    if comm.get_rank() == 0:
                        logger.warning("re-shuffle the dataset")
                    _dataset.shuffle()

            # test
            batch_size = solver.batch_size
            solver.batch_size = 1
            solver.load("model_epoch_%d.pth" % best_epoch)

            if cfg.get("precise_bn", False):
                solver.evaluate("valid", precise_bn=True)
            solver.evaluate("valid")

        if comm.get_rank() == 0:
            logger.warning(">>>>>>>>>> Job %d / %d end >>>>>>>>>>" % (job_id, len(cfgs)))
