#!/usr/bin/env python3
# =============================================================================
import sys, os, argparse, copy
from concurrent.futures import ProcessPoolExecutor

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path

from heuristics import *
from multi_train.envs import InstanceParser

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("domain", help="name of the domain, e.g. navigation")
	parser.add_argument("instance", help="instance number")
	parser.add_argument("-b", "--benchmark", help="folder with domains",
		default="benchmarks")
	parser.add_argument("-l", "--dataset", help="folder with prost logs",
		default="data")
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=["lmc"])
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	if args.domain == "test":
		# Sanity test
		problem = "navigation2x2-inst"
		index_map = ["at11", "at12", "at21", "at22"]
		states = ["1,0,0,0", "0,1,0,0", "0,0,1,0", "0,0,0,1"]
		# Compute heuristics
		planner_exts = PlannerExtensions([problem + ".ppddl"], problem, args.heuristics)
		results = compute_all_heuristics(states, args.heuristics, planner_exts, index_map)
		# Write results
		print(results)
	else:
		problem = f'{args.domain}_inst_mdp__{args.instance}'
		ppddl_file = os.path.join(args.benchmark, args.domain, "ppddl", problem + ".ppddl")
		data_file = os.path.join(args.dataset, "datasets", args.domain, args.instance + ".csv")
		save_file = os.path.join(args.dataset, "heuristics", args.domain, args.instance + ".csv")
		print("Loading " + ppddl_file + "...", flush=True)
		
		planner_exts = PlannerExtensions([ppddl_file], problem, args.heuristics)
		#planner_exts = get_planner_wrapper(ppddl_file, problem, args.heuristics).server.service

		# RDDL parser
		print("Parsing " + problem + "...", flush=True)
		instance_parser = InstanceParser(args.domain, args.instance,
			heuristics=args.heuristics, benchmark_folder=os.path.abspath(args.benchmark))

		# States
		states = read_prost_states(data_file)
		# Compute heuristic
		print("Computing heuristics for " + problem + "...", flush=True)
		results = compute_all_heuristics(states, args.heuristics, planner_exts, instance_parser.num_to_state)	
		# Write results
		write_heuristic_values(save_file, results, instance_parser)