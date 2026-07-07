import sys, os, argparse
import ssipp_interface
import pandas as pd
import numpy as np

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path

from multi_train.deep_plan import my_config
from gym.envs.rddl import instance_parser


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("domain", help="name of the domain, e.g. navigation")
	parser.add_argument("instance", help="number of instance, e.g. 1")
	parser.add_argument("-b", "--benchmark", help="path of prost logs",
		default="benchmarks")
	parser.add_argument("-l", "--dataset", help="path of prost logs",
		default="data")
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=["lmc"])
	return parser.parse_args()


def compute_heuristics(states, heuristic_names, planner_exts, index_map):
	def convert(s):
		return ssipp_interface.convert_symnet_state(s, index_map)
	results = dict()
	for s in states:
		if s in results:
			continue
		state = np.array(s.split(","), dtype="float32")
		values = planner_exts.compute_heuristics(convert(state))
		heuristics = [name + "," + str(h) for name, h in zip(heuristic_names, values)]
		results[s] = s + ":" + ",".join(heuristics) + "\n"
	return results

if __name__ == '__main__':
	args = parse_arguments()
	if args.domain == "test":
		# Sanity test
		problem = "navigation2x2-inst"
		index_map = ["at11", "at12", "at21", "at22"]
		states=["1,0,0,0", "0,1,0,0", "0,0,1,0", "0,0,0,1"]
		# Compute heuristics
		planner_exts = ssipp_interface.PlannerExtensions([problem + ".ppddl"], problem, args.heuristics)
		results = compute_heuristics(states, args.heuristics, planner_exts, index_map)
		# Write results
		print(results)
	else:
		# Files
		problem = f'{args.domain}_inst_mdp__{args.instance}'
		ppddl_file = os.path.join(args.benchmark, args.domain, "ppddl", problem + ".ppddl")
		data_file = os.path.join(args.dataset, "datasets", args.domain, args.instance + ".csv")
		save_file = os.path.join(args.dataset, "heuristics", args.domain, args.instance + ".csv")
		planner_exts = ssipp_interface.PlannerExtensions([ppddl_file], problem, args.heuristics)
		# RDDL parser
		my_config.heuristics = ",".join(args.heuristics)
		my_config.benchmark_folder = os.path.abspath(args.benchmark)
		instance_parser.setup(my_config)
		index_map = instance_parser.InstanceParser(args.domain, args.instance).num_to_state
		# States
		df = pd.read_csv(data_file, delimiter=":", header=None, nrows=None)
		# Compute heuristic
		results = compute_heuristics(df[1], args.heuristics, planner_exts, index_map)	
		# Write results
		with open(save_file, "w") as f:
			f.writelines(results.values())