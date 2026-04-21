import sys, os, io, ast, argparse
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
    parser.add_argument("domain", help="Domain name.")
    parser.add_argument("--train", action="store_true", help="Train before testing.")
    parser.add_argument("--ckpt", type=int, help="Model checkpoint", default=None)
    parser.add_argument("--epochs", type=int, help="Train epochs", default=200)
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
    my_config.trained_model_path = os.path.join(my_config.model_dir, f'{my_config.domain}_{my_config.exp_description}')
    my_config.train_epochs = 200
    if args.ckpt is None:
    	ckpt = 0
    	path = Path(os.path.join(my_config.trained_model_path, "checkpoints"))
    	if path.exists():
    		for file in path.glob('*.index'):
    			i = int(file.stem.replace("ckpt-", ""))
    			if i > ckpt:
    				ckpt = i
    	my_config.exact_checkpoint = str(ckpt)
    else:
    	my_config.exact_checkpoint = str(args.ckpt)

    if args.train:
        import train
        my_config.use_pretrained = False if my_config.exact_checkpoint == "0" else True
        train.train()

    my_config.train_instance = ""
    print("Exact checkpoint:", my_config.exact_checkpoint)
    test.test()
