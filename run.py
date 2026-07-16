import sys, os, io, ast, argparse, subprocess
from pathlib import Path
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
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=[])
	parser.add_argument("-q", "--quick", help="for quick tests (uses fewer instances)", action="store_true")
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

def get_env_var(name):
	try:
		return os.environ[name]
	except KeyError:
		return ""

if __name__ == "__main__":
	args = parse_arguments()
	if args.quick:
		train_instances = "[251, 252, 253]"
		val_instances = "[251, 254]"
		test_instances = "[251, 254]"
	elif my_config.setting == "lr":
		train_instances = "range(1, 1001)"
		val_instances = "range(101, 1101)"
		test_instances = "range(1101, 1301)"
	else: # IPPC
		train_instances = "range(1, 201)"
		val_instances = "range(201, 211)"
		test_instances = "range(211, 251)"

	with open("temp_config.py", "w") as file:
		file.write("domain = '" + args.domain + "'")
		if args.epochs:
			file.write(f"\ntest_instance = ','.join([str(i) for i in {val_instances}])")
			file.write(f"\ntrain_instance = ','.join([str(i) for i in {train_instances}])")
			file.write(f"\ntrain_epochs = {args.epochs}")
		else:
			file.write(f"\ntest_instance = ','.join([str(i) for i in {test_instances}])")
		file.write(f"\nuse_pretrained = {args.restore}")
		file.write(f"\nexp_description = '{args.model}'")
		ckpt = get_last_checkpoint(args.domain, args.model)
		if ckpt >= 0:
			file.write(f"\nexact_checkpoint = '{ckpt}'")
		if args.heuristics:
			heuristics = ",".join([f"'{h}'" for h in args.heuristics])
			print("Using heuristics: " + heuristics)
			file.write(f"\nheuristics = [{heuristics}]")

	root = "/mnt" if get_env_var("IS_WSL") == "true" else os.path.expanduser("~")
	cwd = os.path.abspath("multi_train/deep_plan/")
	cmd = ["podman", "run",
		"--rm", "--env-host", 
		"-v", root + ":" + root,
		"-w", cwd,
		"--device", '"nvidia.com/gpu=all"',
		"--userns=keep-id",
		"symnet-env"]
	config_path = os.path.abspath("temp_config.py")
	if args.epochs:
		cmd += ["python3", "train.py", config_path]
	else:
		cmd += ["python3", "test.py", config_path]
	try:
		process = subprocess.run(cmd)
		#print(process.stdout)
	except subprocess.CalledProcessError as e:
		print(f"Command failed with exit code {e.returncode}")
		print(e.stderr)