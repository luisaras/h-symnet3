import sys, os, argparse, copy
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path

from heuristics import ssipp_interface, problem_server
from multi_train.deep_plan import my_config
from gym.envs.rddl import instance_parser


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


def convert_symnet_state(state, var_names):
	"""Converts state dict to a string format that SSiPP can read.
	var_names should convert an index to a RDDL fluent string."""

	# Prop format: "fluent_name arg1 arg2 argN" 
	format_props = []
	for i, val in enumerate(state):
		if val == 1:
			# format: fluent_name(arg1,args2,argN)
			var = var_names[i].replace("(", " ").replace(")", "").replace(",", " ")
			format_props.append(var)
	format_props.sort()
	return ', '.join(format_props)

def convert_prost_state(state, var_names):
	s = [float(i) for i in state.split(",")]
	return convert_symnet_state(s, var_names)


class PlannerWrapper:
	def __init__(self, server=None, server_args=None):
		self.server = server
		self.server_args = server_args
		self.normalization = 'horizon'
		self.dataset = dict()

	def add_cache(self, file):
		self.h_max = [0] * len(self.instance_parser.heuristic_names)
		self.h_min = [float("inf")] * len(self.instance_parser.heuristic_names)
		with open(file, "r") as f:
			for line in f:
				row = line.split(":")
				values = row[1].strip().split(",")
				h = dict()
				for name, v in zip(values, values[1:]):
					h[name] = float(v)
				atoms = convert_prost_state(row[0].strip(), self.instance_parser.num_to_state)
				values = [h[name] for name in self.instance_parser.heuristic_names]
				self.dataset[atoms] = values
				for i in range(len(values)):
					if self.normalization == 'horizon':
						values[i] /= self.instance_parser.horizon
					elif self.normalization == 'std':
						self.h_max[i] = max(self.h_max[i], values[i])
						self.h_min[i] = min(self.h_min[i], values[i])
		self.null_heuristics = [0] * self.instance_parser.get_num_heuristics()
		if self.normalization == 'std':
			for values in self.dataset.values():
				for i in range(len(values)):
					n = self.h_max[i] - self.h_min[i]
					if n > 0:
						values[i] = (values[i] - self.h_min[i]) / n 


	def compute_heuristics(self, state):
		atoms = convert_symnet_state(state, self.instance_parser.num_to_state)
		if atoms in self.dataset:
			return self.dataset[atoms]
		if self.server is None:
			if self.server_args is None:
				return self.null_heuristics
			args = self.server_args
			# Build server on demand
			self.server = problem_server.make_planner_server(*args)
			self.server_args = None
		# Compute on the fly
		heuristics = self.server.service.compute_heuristics(atoms)
		if self.normalization == 'std':
			for i in range(len(heuristics)):
				n = self.h_max[i] - self.h_min[i]
				if n > 0:
					heuristics[i] = (heuristics[i] - self.h_min[i]) / n 
		elif self.normalization == 'horizon':
			for i in range(len(heuristics)):
				heuristics[i] /= self.instance_parser.horizon
		self.dataset[atoms] = heuristics
		return heuristics


wrappers = dict()
wrapper_type = "on_demand"
def get_planner_wrapper(ppddl_file, instance_name, heuristics):
	if instance_name in wrappers:
		return wrappers[instance_name]
	else:
		if wrapper_type == "null":
			wrapper = PlannerWrapper()
		elif wrapper_type == "on_demand":
			wrapper = PlannerWrapper(server_args=(ppddl_file, instance_name, heuristics))
		else:
			server = problem_server.make_planner_server(ppddl_file, instance_name, heuristics)
			wrapper = PlannerWrapper(server=server)
		wrappers[instance_name] = wrapper
		return wrapper


def compute_heuristics(states, heuristic_names, planner_exts, index_map):
	results = dict()
	for s in states:
		if s in results:
			continue
		atoms = convert_prost_state(s, index_map)
		print("Heuristics for state: " + str(atoms))
		values = planner_exts.compute_heuristics(atoms)
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
		problem = f'{args.domain}_inst_mdp__{args.instance}'
		ppddl_file = os.path.join(args.benchmark, args.domain, "ppddl", problem + ".ppddl")
		data_file = os.path.join(args.dataset, "datasets", args.domain, args.instance + ".csv")
		save_file = os.path.join(args.dataset, "heuristics", args.domain, args.instance + ".csv")
		print("Loading " + ppddl_file + "...", flush=True)
		
		#planner_exts = ssipp_interface.PlannerExtensions([ppddl_file], problem, args.heuristics)
		planner_exts = get_planner_wrapper(ppddl_file, problem, args.heuristics).server.service

		# RDDL parser
		print("Parsing " + problem + "...", flush=True)
		my_config.heuristics = args.heuristics
		my_config.benchmark_folder = os.path.abspath(args.benchmark)
		instance_parser.setup(my_config)
		index_map = instance_parser.InstanceParser(args.domain, args.instance).num_to_state

		# States
		df = pd.read_csv(data_file, delimiter=":", header=None, nrows=None)
		# Compute heuristic
		print("Computing heuristics for " + problem + "...", flush=True)
		results = compute_heuristics(df[1], args.heuristics, planner_exts, index_map)	
		# Write results
		with open(save_file, "w") as f:
			f.writelines(results.values())