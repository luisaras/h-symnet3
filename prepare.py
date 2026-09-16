#!/usr/bin/env python3
# =============================================================================

import sys, os, io, ast, argparse, subprocess, copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from run import run_symnet3_env, get_setting_instances
from benchmarks.instance_generator import Generator
from data.dataset_builder import create_dataset

IPPC=["academic_advising_ippc", "crossing_traffic", "game_of_life", "navigation", "skill_teaching", "sysadmin", "tamarisk", "traffic", "wildfire"]
LR=["academic_advising_chain", "academic_advising", "pizza_delivery", "pizza_delivery_grid", "pizza_delivery_windy", "wall", "stochastic_navigation", "stochastic_wall corridor"]
EXTRA=["recon", "triangle_tireworld", "elevators"]
TEST=["navigation", "academic_advising", "exploding_blocks"]
DET=["navigation"]

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
	parser.add_argument("-s", "--setting",  help='"debug": for quick sanity tests (uses debug instances)\n' +
												 '"local": for local tests (uses fewer instances)' +
												 '"lr": long-range instances' +
												 '"ippc": ippc instances' +
												 '"det": deterministic instances',
		default="",
		choices=["debug", "ippc1", "ippc2", "ippc3", "det", "lr", "extra", ""])
	args = parser.parse_args()
	args.run_all = not (args.generate or args.parse or args.plan or args.heuristics)
	return args


def generate_instances(first_inst, last_inst, domains, setting, skip=False):
	print("Generating RDDL instances.")
	generator = Generator(None, verbose=True)
	def generate(f, l, dataset):
		f = max(f, first_inst)
		l = min(l, last_inst+1)
		if l > f:
			for d in domains:
				generator.generate_all(d, dataset, range(f, l), seed=-1, skip=skip)
	if setting == "debug" or setting == "":
		generate(251, 252, "debug1")
		generate(252, 253, "debug2")
		generate(253, 254, "debug3")
		generate(254, 255, "debug4")
	if setting != "debug":
		if setting == "":
			setting = "ippc3"
		i, _ = get_setting_instances(setting)
		if "ippc" in setting:
			setting = "ippc"
		generate(i[0], i[1], setting + "-train")
		generate(i[1], i[2], setting + "-val")
		generate(i[2], i[3], setting + "-test")


def preprocess_rddl(first_inst, last_inst, domains, skip=False):
	print("Generate DBN files.")
	cmd = ["python3", "parse_instances.py"] + domains
	cmd += ["-i", str(first_inst), str(last_inst)]
	if skip:
		cmd += ["--skip"]
	run_symnet3_env(cmd)


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


def precompute_heuristics(first_inst, last_inst, domains, skip=False):
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
	domains = args.domains
	setting = args.setting.lower()
	if len(args.domains) == 0:
		if setting == "ippc":
			domains = IPPC
		elif setting == "lr":
			domains = LR
		elif setting == "extra":
			domains = EXTRA
		elif setting == "det":
			domains = DET
		else: # debug or local
			domains = TEST

	first_inst = args.ins[0]
	last_inst = args.ins[1]
	if setting != "":
		i = get_setting_instances(setting)
		first_inst = max(first_inst, i[0])
		last_inst = min(last_inst, i[3]-1)

	print(f"Preparing instances {first_inst} to {last_inst} of domains: {",".join(domains)}" )
	if args.generate:
		generate_instances(first_inst, last_inst, domains, setting)
	if args.parse:
		preprocess_rddl(first_inst, last_inst, domains, skip=False)
	if args.plan:
		generate_trajectories(first_inst, last_inst, domains, skip=False)
	if args.heuristics:
		precompute_heuristics(first_inst, last_inst, domains, skip=False)
	if args.run_all:
		generate_instances(first_inst, last_inst, domains, setting)
		preprocess_rddl(first_inst, last_inst, domains, skip=True)
		generate_trajectories(first_inst, last_inst, domains, skip=True)
		precompute_heuristics(first_inst, last_inst, domains, skip=True)
