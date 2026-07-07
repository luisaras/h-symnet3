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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("domain", help="name of the domain, e.g. navigation")
	parser.add_argument("instance", help="number of instance, e.g. 1")
	parser.add_argument("-b", "--benchmark", help="path of prost logs",
		default="benchmarks")
	parser.add_argument("-l", "--dataset", help="path of prost logs",
		default="data")
	parser.add_argument("-f", "--heuristics", help="heuristic features (lmc, hadd, hmax)",
		nargs="*", default=["lmc"])
	args = parser.parse_args()

	problem = f'{args.domain}_inst_mdp__{args.instance}'

	ppddl_file = os.path.join(args.benchmark, args.domain, "ppddl", problem + ".ppddl")
	data_file = os.path.join(args.dataset, "datasets", args.domain, args.instance + ".csv")
	save_file = os.path.join(args.dataset, "heuristics", args.domain, args.instance + ".csv")

	my_config.heuristics = ",".join(args.heuristics)
	my_config.benchmark_folder = os.path.abspath(args.benchmark)
	instance_parser.setup(my_config)
	ip = instance_parser.InstanceParser(args.domain, args.instance)
	planner_exts = ssipp_interface.PlannerExtensions(ppddl_file, problem, args.heuristics)

	results = dict()
	df = pd.read_csv(data_file, delimiter=":", header=None, nrows=None)
	for s in df[1]:
		if s in results:
			continue
		state = np.array(s.split(","), dtype="float32")
		values = planner_exts.compute_heuristics(state, ip)
		heuristics = [name + "," + str(h) for name, h in zip(args.heuristics, values)]
		results[s] = s + ":" + ",".join(heuristics) + "\n"

	with open(save_file, "w") as f:
		f.writelines(results.values())