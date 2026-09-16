import sys, os
from .ssipp_interface import num_heuristic_features
from .problem_server import make_planner_server


wrappers = dict()
WRAPPER_TYPE = "start"
NORMALIZATION = "none"
def setup(wrapper_type, normalization):
	global WRAPPER_TYPE
	WRAPPER_TYPE = wrapper_type
	global NORMALIZATION
	NORMALIZATION = normalization


def get_planner_wrapper(ppddl_file, instance_name, heuristics):
	if instance_name in wrappers:
		return wrappers[instance_name]
	else:
		if WRAPPER_TYPE == "null":
			wrapper = PlannerWrapper(instance_name, heuristics)
		elif WRAPPER_TYPE == "on_demand":
			wrapper = PlannerWrapper(instance_name, heuristics, ppddl_file=ppddl_file)
		else:
			server = make_planner_server(ppddl_file, instance_name, heuristics)
			wrapper = PlannerWrapper(instance_name, heuristics, server=server)
		wrappers[instance_name] = wrapper
		return wrapper


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

class PlannerWrapper:
	def __init__(self, problem, heuristic_names, server=None, ppddl_file=None):
		self.problem = problem
		self.server = server
		self.ppddl_file = ppddl_file
		self.heuristic_names = heuristic_names
		self._cache = dict()
		self.h_max = None
		self.h_min = None

	def get_num_heuristic_features(self):
		return num_heuristic_features(self.heuristic_names)

	def add_cache(self, file):
		dim = self.get_num_heuristic_features()
		self.null_heuristics = [0] * dim
		dataset = read_heuristic_values(file, self.instance_parser)
		for s, heuristics in list(dataset.items()):
			dataset[s] = merge_heuristics(heuristics, self.heuristic_names)
		self._cache.update(dataset)
		if NORMALIZATION == 'max':
			if self.h_max is None:
				self.h_max = [0] * dim
				self.h_min = [float("inf")] * dim
			self.update_min_max(dataset)
			for values in dataset.values():
				self.normalize_min_max(values)
		elif NORMALIZATION == 'horizon':
			for values in dataset.values():
				for i in range(len(values)):
					values[i] /= self.instance_parser.horizon

	def update_min_max(self, dataset):
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
			if self.ppddl_file is None:
				return self.null_heuristics
			args = (self.ppddl_file, self.problem, self.heuristic_names)
			print("Building server for problem " + self.problem + " on demand to compute state: " + atoms)
			self.server = make_planner_server(*args)
		# Compute on the fly
		heuristics = self.server.service.compute_heuristics(atoms)
		features = merge_heuristics(heuristics)
		if NORMALIZATION == 'max':
			self.normalize_min_max(features)
		elif NORMALIZATION == 'horizon':
			for i in range(len(features)):
				features[i] /= self.instance_parser.horizon
		self._cache[atoms] = features
		return features
