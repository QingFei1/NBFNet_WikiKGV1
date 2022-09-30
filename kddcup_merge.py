import os
import glob

import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="directory to dump files")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    range2pred = {}
    file_names = glob.glob(os.path.join(args.path, "t_pred_wikikg90m_*_*.npz"))
    for file_name in file_names:
        print("loading `%s`" % file_name)
        dump = np.load(file_name)
        start, end = dump["test_range"]
        pred = dump["t_pred_top10"]
        range2pred[(start, end)] = pred

    preds = []
    last = -1
    for start, end in sorted(range2pred.keys()):
        preds.append(range2pred[(start, end)])
        if last != -1 and start != last:
            raise ValueError("Prediction dumps are not contiguous")
        last = end
    pred = np.concatenate(preds, axis=0)
    print("merge done!")
    print("output shape: %s, dtype: %s" % (pred.shape, pred.dtype))
    save_file = os.path.join(args.path, "t_pred_wikikg90m")
    np.savez_compressed(save_file, t_pred_top10=pred)