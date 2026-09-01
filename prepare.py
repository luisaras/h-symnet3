#!/usr/bin/env python3
# =============================================================================

import sys, os, io, ast, argparse, subprocess, copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from run import symnet3_env_cmd, run_symnet3_env
from benchmarks.generator import Generator
from prost.dataset_builder import create_dataset

IPPC=["academic_advising_ippc", "crossing_traffic", "game_of_life", "navigation", "skill_teaching", "sysadmin", "tamarisk", "traffic", "wildfire"]
LR=["academic_advising_chain", "academic_advising", "pizza_delivery", "pizza_delivery_grid", "pizza_delivery_windy", "wall", "stochastic_navigation", "stochastic_wall corridor"]
EXTRA=["recon", "triangle_tireworld", "elevators"]
TEST=["navigation, academic_advising, exploding_blocks"]

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Prepare benchmark and datasets for SymNet. "
		"Run this ONLY after running install.sh.\n"
		"If no flag is set, it will generate, parse, plan and compute heuristics for any missing instances.",
		formatter_class=formatter,
	)
	parser.add_argument("domains", help="domain name(s)",
		nargs="*", default=TEST)
	parser.add_argument("--generate", help="generate RDDL and PPDDL instance files",
		action="store_true")
	parser.add_argument("--parse", help="generate DBN files for the RDDL instances",
		action="store_true")
	parser.add_argument("--plan", help="run PROST and collect trajectories",
		action="store_true")
	parser.add_argument("--heuristics", help="compute heuristics for PROST states using PPDDL files",
		action="store_true")
	parser.add_argument("-i", "--ins", help="first and last instances", 
		nargs=2, type=int, default=[1, 254])
	args = parser.parse_args()
	args.run_all = not (args.generate or args.parse or args.plan or args.heuristics)
	return args


def generate_instances(first_inst, last_inst, domains, skip=False):
	print("Generating RDDL instances.")
	generator = Generator(None, verbose=True)
	def generate(d, f, l, dataset):
		f = max(f, first_inst)
		l = min(l, last_inst)
		n = l - f + 1
		if n > 0:
			generator.generate_all(d, dataset, f, n, seed=-1, skip=skip)
	for d in domains:
		generate(d, 1, 200, "train")
		generate(d, 201, 210, "val")
		generate(d, 211, 250, "test")
		generate(d, 251, 251, "debug1")
		generate(d, 252, 252, "debug2")
		generate(d, 253, 253, "debug3")
		generate(d, 254, 254, "debug4")


def preprocess_rddl(first_inst, last_inst, domains, skip=False):
	print("Generate DBN files.")
	arg = "skip" if skip else ""
	instances = [str(i) for i in range(first_inst, last_inst+1)]
	script = f"""
	echo $(pwd)
	for d in {" ".join(domains)}; do
		for i in {" ".join(instances)}; do
			./parse_instance.sh $d $i {arg}
		done
	done
	"""
	run_symnet3_env(["bash", "-c", script])


def generate_trajectories(first_inst, last_inst, domains, skip=False):
	os.makedirs("data/datasets", exist_ok=True)
	os.makedirs("data/logs", exist_ok=True)
	cmd = ["python3", "prost/run_prost.py"] + domains
	cmd += ["-i", str(first_inst), str(last_inst)]
	cmd += ["-d", "benchmarks/{domain}/rddl"]
	cmd += ["-l", "data/logs/{domain}"]
	cmd += ["-w", "8"]
	if skip:
		cmd += ["--skip"]
	run_symnet3_env(cmd)
	for d in domains:
		prost_log = "data/logs/" + d
		save_folder = "data/datasets/" + d
		create_dataset(d, first_inst, last_inst, prost_log, save_folder)


def compute_heuristics(first_inst, last_inst, domains, skip):
	for d in domains:
		save_folder = os.path.join("data", "heuristics", d)
		os.makedirs("data/heuristics/" + d, exist_ok=True)
		print("Computing heuristics for " + d + "...")
		cmd = ["python3", "heuristics/compute_heuristics.py", d, ""]
		for i in range(first_inst, last_inst+1):
			if skip and os.path.exists(os.path.join(save_folder, f"{i}.csv")):
				print(f"Skipped instance {i}")
				continue
			cmd[3] = str(i)
			run_symnet3_env(cmd)


if __name__ == "__main__":
	args = parse_arguments()
	domains = TEST
	if len(args.domains) > 0:
		if args.domains[0].upper() == "IPPC":
			domains = IPPC
		elif args.domains[0].upper() == "LR":
			domains = LR
		elif args.domains[0].upper() == "EXTRA":
			domains = EXTRA
		else:
			domains = args.domains
	print(f"Preparing instances {args.ins[0]} to {args.ins[1]} of domains: {",".join(domains)}" )
	if args.generate:
		generate_instances(args.ins[0], args.ins[1], domains)
	if args.parse:
		preprocess_rddl(args.ins[0], args.ins[1], domains, skip=False)
	if args.plan:
		generate_trajectories(args.ins[0], args.ins[1], domains, skip=False)
	if args.heuristics: 
		compute_heuristics(args.ins[0], args.ins[1], domains, skip=False)
	if args.run_all:
		generate_instances(args.ins[0], args.ins[1], domains)
		preprocess_rddl(args.ins[0], args.ins[1], domains, skip=True)
		generate_trajectories(args.ins[0], args.ins[1], domains, skip=True)
		compute_heuristics(args.ins[0], args.ins[1], domains, skip=True)
