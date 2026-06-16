import sys, os, io, ast, argparse, subprocess
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout
from multi_train.deep_plan import my_config

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
    parser.add_argument("-f", "--heuristics", help="heuristic features",
        nargs="*", default=[])
    args = parser.parse_args()
    return args

def get_last_checkpoint(domain, exp_description):
    model_suffix = f"{domain}_{exp_description}"
    MODEL_DIR = os.path.join("multi_train", "deep_plan", my_config.model_dir, model_suffix, "checkpoints") 
    last = -1
    for file in Path(os.path.abspath(MODEL_DIR)).glob("ckpt-*.index"):
        i = int(file.stem.replace("ckpt-", "").replace(".index", ""))
        if i > last:
            last = i
    return last

if __name__ == "__main__":
    args = parse_arguments()

    with open("temp_config.py", "w") as file:
        file.write("domain = '" + args.domain + "'")
        file.write("\ntest_instance = ','.join([str(i+1) for i in range(10)])")
        if args.epochs:
            file.write("\ntrain_instance = ','.join([str(i+1) for i in range(3)])")
            file.write("\ntrain_epochs = " + str(args.epochs))
        file.write("\nuse_pretrained = " + str(args.restore))
        file.write("\nexp_description = '" + args.model + "'")
        ckpt = get_last_checkpoint(args.domain, args.model)
        if ckpt >= 0:
            file.write("\nexact_checkpoint = '" + str(ckpt) + "'")
        file.write("\nkeep_ckpts = True")
        if args.heuristics:
            heuristics = ','.join(args.heuristics)
            file.write("\nheuristics = '" + heuristics + "'")

    config_path = os.path.abspath("temp_config.py")
    if args.epochs:
        cmd = ["python3", "train.py", config_path]
    else:
        cmd = ["python3", "test.py", config_path]
   
    try:
        process = subprocess.run(cmd, cwd="multi_train/deep_plan/")
        #print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(e.stderr)