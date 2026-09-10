import sys, os, argparse, copy
from concurrent.futures import ProcessPoolExecutor

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


def symnet2ssipp_state(state: list, var_names):
	"""Converts state list to a string format that SSiPP can read.
	var_names should convert an index to a RDDL fluent string."""

	# Prop format: "fluent_name arg1 arg2 argN" 
	format_props = []
	for i, val in enumerate(state):
		if val == 1 and var_names[i] != "termination":
			# format: fluent_name(arg1,args2,argN)
			var = var_names[i].replace("(", " ").replace(")", "").replace(",", " ")
			format_props.append(var)
	format_props.sort()
	return ', '.join(format_props)


def prost2ssipp_state(state, var_names):
	s = [float(i) for i in state.split(",")]
	return symnet2ssipp_state(s, var_names)


def merge_heuristics(heuristics, heuristic_names=None):
	features = []
	if heuristic_names is None: # list of lists
		for h in heuristics:
			features.extend(h)
	else:
		for name in heuristic_names: # dict of lists
			features.extend(heuristics[name])
	return features

def read_heuristic_values(file, instance_parser) -> dict:
	dataset = dict()
	with open(file, "r") as f:
		for line in f:
			row = line.split(":")
			atoms = prost2ssipp_state(row.pop(0), instance_parser.num_to_state)
			heuristics = dict()
			for i, h in enumerate(row):
				values = h.strip().split(",")
				name = values.pop(0)
				heuristics[name] = [float(v) for v in values]
			dataset[atoms] = heuristics
	return dataset

def write_heuristic_values(file, dataset: dict, instance_parser):
	# Compute heuristic
	print("Computing heuristics for " + problem + "...", flush=True)
	# Write results
	with open(file, "w") as f:
		for s, values in dataset.items():
			heuristics = [",".join([name] + list(map(str, h))) for name, h in zip(instance_parser.heuristic_names, values)]
			f.write(":".join([s] + heuristics) + "\n")

def read_prost_states(file):
	states = []
	with open(file, "r") as f:
		for line in f:
			row = line.split(":")
			states.append(row[1])
	return states


class PlannerWrapper:
	def __init__(self, problem, server=None, server_args=None):
		self.problem = problem
		self.server = server
		self.server_args = server_args
		self.normalization = 'horizon'
		self._cache = dict()
		self.h_max = None
		self.h_min = None

	def get_num_heuristic_features(self):
		return ssipp_interface.num_heuristic_features(self.instance_parser.heuristic_names)

	def add_cache(self, file):
		self.null_heuristics = [0] * self.get_num_heuristic_features()
		dataset = read_heuristic_values(file, self.instance_parser)
		for s, heuristics in list(dataset.items()):
			dataset[s] = merge_heuristics(heuristics, self.instance_parser.heuristic_names)
		self._cache.update(dataset)
		if self.normalization == 'max':
			if self.h_max is None:
				self.h_max = [0] * dim
				self.h_min = [float("inf")] * dim
			self.update_min_max(dataset)
			for values in dataset.values():
				self.normalize_min_max(values)
		elif self.normalization == 'horizon':
			for values in dataset.values():
				for i in range(len(values)):
					values[i] /= self.instance_parser.horizon

	def update_min_max(dataset):
		dim = self.get_num_heuristic_features()
		for values in dataset.values():
			for i in range(dim):
				self.h_max[i] = max(self.h_max[i], values[i])
				self.h_min[i] = min(self.h_min[i], values[i])

	def normalize_min_max(self, values):		
		for i in range(len(values)):
			n = self.h_max[i] - self.h_min[i]
			if n > 0:
				values[i] = (values[i] - self.h_min[i]) / n 

	def get_heuristics(self, state: list) -> list:
		atoms = symnet2ssipp_state(state, self.instance_parser.num_to_state)
		if atoms in self._cache:
			return self._cache[atoms]
		if self.server is None:
			if self.server_args is None:
				return self.null_heuristics
			args = self.server_args
			print("Building server for problem " + self.problem + " on demand to compute state: " + atoms)
			self.server = problem_server.make_planner_server(*args)
			self.server_args = None
		# Compute on the fly
		heuristics = self.server.service.compute_heuristics(atoms)
		features = merge_heuristics(heuristics)
		if self.normalization == 'max':
			self.normalize_min_max(features)
		elif self.normalization == 'horizon':
			for i in range(len(features)):
				features[i] /= self.instance_parser.horizon
		self._cache[atoms] = features
		return features


wrappers = dict()
wrapper_type = "start"
def get_planner_wrapper(ppddl_file, instance_name, heuristics):
	if instance_name in wrappers:
		return wrappers[instance_name]
	else:
		if wrapper_type == "null":
			wrapper = PlannerWrapper(instance_name)
		elif wrapper_type == "on_demand":
			wrapper = PlannerWrapper(instance_name,server_args=(ppddl_file, instance_name, heuristics))
		else:
			server = problem_server.make_planner_server(ppddl_file, instance_name, heuristics)
			wrapper = PlannerWrapper(instance_name, server=server)
		wrappers[instance_name] = wrapper
		return wrapper


def compute_all_heuristics(states, heuristic_names, planner_exts, index_map) -> dict:
	results = dict()
	for s in states:
		if s in results:
			continue
		atoms = prost2ssipp_state(s, index_map)
		print("Heuristics for state: " + str(atoms))
		values = planner_exts.compute_heuristics(atoms)
		if values is None:
			print("Failed to compute heuristics")
			continue
		results[s] = values
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
		results = compute_all_heuristics(states, args.heuristics, planner_exts, index_map)
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
		instance_parser = instance_parser.InstanceParser(args.domain, args.instance)

		# States
		states = read_prost_states(data_file)
		# Compute heuristic
		print("Computing heuristics for " + problem + "...", flush=True)
		results = compute_all_heuristics(states, args.heuristics, planner_exts, instance_parser.num_to_state)	
		# Write results
		write_heuristic_values(save_file, results, instance_parser)