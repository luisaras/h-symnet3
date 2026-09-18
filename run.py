#!/usr/bin/env python3
# =============================================================================

import sys, os, io, ast, argparse, subprocess
from pathlib import Path

MODEL_DIR = os.path.join("multi_train", "supervised", "models") 

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Run SymNet3.0 in the directory defined "
		"by environment variable SYMNET_ROOT.",
		formatter_class=formatter,
	)
	parser.add_argument("domain", help="domain name")
	parser.add_argument("-m", "--model", help="model type",
		default="standard")
	parser.add_argument("-d", "--model_dir", help="model directory",
		default=MODEL_DIR)
	parser.add_argument("-e", "--epochs", help="train epochs (if none, it will only test trained models)", 
		type=int, default=None)
	parser.add_argument("-r", "--restore", help="load from (specified or last) checkpoint instead of training from scratch",
		nargs="?", default="", const="-1")
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=[])
	parser.add_argument("-n", "--n_samples", help="number of successor state samples",
		type=int, default=1)
	parser.add_argument("-s", "--setting",  help='"debug": for quick sanity tests (uses debug instances)\n' +
												 '"local": for local tests (uses fewer instances)' +
												 '"lr": long-range instances' +
												 '"ippc": ippc instances' +
												 '"det": deterministic instances',
		default="ippc3",
		choices=["debug", "ippc1", "ippc2", "ippc3", "det", "lr", "extra"])
	parser.add_argument("-l", "--local", help="",
		action="store_true")
	parser.add_argument("--lr",          help="for long-range tests",
		action="store_true")
	parser.add_argument("--ippc",          help="for long-range tests",
		action="store_true")
	args = parser.parse_args()
	return args

def get_last_checkpoint(domain, exp_description, model_dir):
	model_name = f"{domain}_{exp_description}"
	model_dir = os.path.join(model_dir, model_name, "checkpoints") 
	last = -1
	for file in Path(os.path.abspath(model_dir)).glob("ckpt-*.index"):
		i = int(file.stem.replace("ckpt-", "").replace(".index", ""))
		if i > last:
			last = i
	return last

def get_env_var(name):
	try:
		return os.environ[name]
	except KeyError:
		return ""

def symnet3_env_cmd(cwd="."):
	cmd = ["podman"]
	if get_env_var("IS_WSL") == "true":
		gpu_flags=["--device", "\"nvidia.com/gpu=all\""]
		root = "/mnt"
	else:
		gpu_flags=["--device", "/dev/nvidia0",
			"--device", "/dev/nvidiactl",
			"--device", "/dev/nvidia-uvm",
			"-v", "/usr/lib/x86_64-linux-gnu/nvidia:/host-nvidia:ro",
			"--env", "LD_LIBRARY_PATH=/host-nvidia"]
		root = os.path.expanduser("~")
#		root = get_env_var("HOME")
#		cmd += ["--cdi-spec-dir=" + root + "/.config/cdi"]
	#cwd = os.path.abspath(".")
	cwd = os.path.abspath(os.path.join(os.environ.get('PWD', os.getcwd()), cwd))
	cmd += ["run", "--rm",
		"--env-host",
		"--userns=keep-id",
		"-v", root + ":" + root,
		"-w", cwd]
	cmd += gpu_flags
	cmd += ["symnet-env"]
	return cmd


def run_symnet3_env(cmd, cwd="."):
	cmd = symnet3_env_cmd(cwd) + cmd
	try:
		#process = subprocess.run(cmd)
		#if process.stdout:
		#	print(process.stdout)
		with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, text=True) as process:
			for line in process.stdout:
				print(line, end="") # 'end=""' avoids adding double newlines
			# The exit code is available after the loop finishes
			return_code = process.wait()
			if return_code != 0:
				print(process.stderr)
	except subprocess.CalledProcessError as e:
		print(e.stdout)
		print(f"Command failed with exit code {e.returncode}")
		print(e.stderr)


def get_setting_instances(setting):
	if setting == "debug":
		return (251, 253, 254, 255)
	elif setting == "ippc1":
		return (1, 4, 5, 11)
	elif setting == "ippc2":
		return (1, 31, 41, 51)
	elif setting == "ippc3":
		return (1, 201, 211, 251)
	elif setting == "lr":
		return (1001, 2001, 2101, 2301)
	elif setting == "det":
		return (255, 455, 465, 505)
	else:
		raise Exception("setting not defined: " + str(setting))


if __name__ == "__main__":
	args = parse_arguments()

	i = get_setting_instances(args.setting)
	args.model = args.setting + "_" + args.model
	train_instances = f"range({i[0]}, {i[1]})"
	val_instances = f"range({i[1]}, {i[2]})"
	test_instances = f"range({i[2]}, {i[3]})"
	if args.n_samples > 1:
		args.model += "-x" + str(args.n_samples)

	with open(os.path.join("multi_train", "temp_config.py"), "w") as file:
		file.write(f"domain = '{args.domain}'\n")
		file.write(f"setting = '{args.setting}'\n")
		file.write(f"model_dir = '{args.model_dir}'\n")
		if args.epochs:
			file.write(f"test_instance = ','.join([str(i) for i in {val_instances}])\n")
			file.write(f"train_instance = ','.join([str(i) for i in {train_instances}])\n")
			file.write(f"train_epochs = {args.epochs}\n")
		else:
			file.write(f"train_instance = ''\n")
			file.write(f"test_instance = ','.join([str(i) for i in {test_instances}])\n")
		file.write(f"exp_description = '{args.model}'\n")
		ckpt = get_last_checkpoint(args.domain, args.model, args.model_dir)
		if args.restore != "":
			file.write(f"use_pretrained = True\n")
			exact_ckpt = int(args.restore)
			if exact_ckpt >= 0:
				ckpt = exact_ckpt
		else:
			file.write(f"use_pretrained = False\n")
		if ckpt >= 0:
			file.write(f"exact_checkpoint = '{ckpt}'\n")
		else:
			file.write(f"exact_checkpoint = None\n")
		if args.heuristics:
			heuristics = ",".join([f"'{h}'" for h in args.heuristics])
			print("Using heuristics: " + heuristics)
			file.write(f"heuristics = [{heuristics}]\n")
			if "cache" in args.model:
				file.write(f"init_heuristics = 'null'\n")
			if "norm0" in args.model:
				file.write(f"heuristic_normalization = 'horizon'\n")
			elif "norm1" in args.model:
				file.write(f"heuristic_normalization = 'max'\n")
		file.write(f"heuristic_samples = {args.n_samples}\n")

	py_dir = os.path.join("multi_train", "supervised")
	config_path = os.path.join("..", "temp_config.py")
	if args.epochs:
		cmd = ["python3", "train.py", config_path]
	else:
		cmd = ["python3", "test.py", config_path]
	run_symnet3_env(cmd, py_dir)
	#os.remove(os.path.join("multi_train", "temp_config.py"))