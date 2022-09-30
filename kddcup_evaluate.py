import os
import glob
import pickle

import argparse
import numpy as np

from ogb import lsc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="prediction dump file")
    parser.add_argument("-d", "--dataset", help="directory to dataset", default="~/projects/rrg-bengioy-ad/graph/dataset")
    parser.add_argument("-i", "--indices", help="dump to valid set indices")
    parser.add_argument("-s", "--start", help="start sample id", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.dataset = os.path.expanduser(args.dataset)
    index_file = os.path.join(args.dataset, "wikikg90m_kddcup2021/processed/val_t_correct_index.npy")
    t_correct_index = np.load(index_file)

    t_pred_top10 = np.load(args.file_name)["t_pred_top10"]
    num_sample = len(t_pred_top10)
    print("evaluate %d predictions" % num_sample)
    input_dict = {}
    if args.indices:
        with open(args.indices, "rb") as fin:
            indices = pickle.load(fin)
        input_dict["h,r->t"] = {"t_pred_top10": t_pred_top10, "t_correct_index": t_correct_index[indices]}
    else:
        input_dict["h,r->t"] = {"t_pred_top10": t_pred_top10, "t_correct_index": t_correct_index[args.start: args.start + num_sample]}

    evaluator = lsc.WikiKG90MEvaluator()
    print("mrr (lsc): %g" % evaluator.eval(input_dict)["mrr"])