import sys, os, io, ast, argparse
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout

def parse_arguments():
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=38
    )
    parser = argparse.ArgumentParser(
        description="Run SymNet3.0 from directory defined "
        "by environment variable SYMNET_ROOT.",
        formatter_class=formatter,
    )
    parser.add_argument("domain", help="domain name")
    parser.add_argument("-m", "--model", help="model type", default="standard")
    parser.add_argument("-e", "--epochs", type=int, help="train epochs", default=None)
    parser.add_argument("-r", "--restore", help="load from checkpoint instead of training from scratch", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    path = os.path.abspath(".") + "/multi_train/deep_plan/"
    sys.path.insert(0, path)
    os.chdir(path)
    import my_config, test

    args = parse_arguments()

    my_config.domain = args.domain
    my_config.train_instance = ",".join([str(i+1) for i in range(3)])
    my_config.test_instance = ",".join([str(i+1) for i in range(10)])
    my_config.exp_description = args.model
    my_config.trained_model_path = os.path.join(my_config.model_dir, f'{my_config.domain}_{my_config.exp_description}')
    my_config.use_pretrained = args.restore

    if args.epochs:
        import train
        my_config.train_epochs = args.epochs
        train.train()
    else:
        my_config.train_instance = ""
        ptr = open(f'{my_config.trained_model_path}/meta_logging.csv')
        ckpt = 1
        best_ckpt, best_rew = 1, -1000000
        for line in ptr.readlines()[2:]:
            if my_config.setting == "lr":
                toks = line.split(",")[:100] # 100 val instances in lr
            else:
                toks = line.split(",")[:10] # 10 val instances in ippc
            rew = np.mean([float(x) for x in toks])
            if rew > best_rew:
                best_rew = rew
                best_ckpt = ckpt
            ckpt += 1
        my_config.exact_checkpoint = str(best_ckpt)
        test.test()
