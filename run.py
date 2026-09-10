#!/usr/bin/env python3
# =============================================================================

import sys, os, io, ast, argparse, subprocess
from pathlib import Path

from multi_train.deep_plan import my_config

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
	parser.add_argument("-e", "--epochs", help="train epochs (if none, it will only test trained models)", 
		type=int, default=None)
	parser.add_argument("-r", "--restore", help="load from (specified or last) checkpoint instead of training from scratch",
		nargs="?", default="", const="-1")
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=[])
	parser.add_argument("-t", "--test", help="for quick sanity tests (uses debug instances)",
		action="store_true")
	parser.add_argument("-l", "--local", help="for local tests (uses fewer instances)",
		action="store_true")
	args = parser.parse_args()
	return args

def get_last_checkpoint(domain, exp_description):
	model_name = f"{domain}_{exp_description}"
	model_dir = os.path.join("multi_train", "deep_plan", my_config.model_dir, model_name, "checkpoints") 
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


if __name__ == "__main__":
	args = parse_arguments()
	if args.test:
		train_instances = "[251, 252, 253]"
		val_instances = "[251, 254]"
		test_instances = "[251, 254]"
		args.model = "test_" + args.model 
	elif args.local:
		train_instances = "range(1, 31)"
		val_instances = "range(31, 41)"
		test_instances = "range(41, 51)"
		args.model = "mini_" + args.model 
	elif my_config.setting == "lr":
		train_instances = "range(1, 1001)"
		val_instances = "range(101, 1101)"
		test_instances = "range(1101, 1301)"
	else: # IPPC
		train_instances = "range(1, 201)"
		val_instances = "range(201, 211)"
		test_instances = "range(211, 251)"

	with open("temp_config.py", "w") as file:
		file.write(f"domain = '{args.domain}'\n")
		if args.epochs:
			file.write(f"test_instance = ','.join([str(i) for i in {val_instances}])\n")
			file.write(f"train_instance = ','.join([str(i) for i in {train_instances}])\n")
			file.write(f"train_epochs = {args.epochs}\n")
		else:
			file.write(f"test_instance = ','.join([str(i) for i in {test_instances}])\n")
		file.write(f"exp_description = '{args.model}'\n")
		ckpt = get_last_checkpoint(args.domain, args.model)
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

	py_dir = os.path.join("multi_train", "deep_plan")
	config_path = os.path.join("..", "..", "temp_config.py")
	if args.epochs:
		cmd = ["python3", "train.py", config_path]
	else:
		cmd = ["python3", "test.py", config_path]
	run_symnet3_env(cmd, py_dir)