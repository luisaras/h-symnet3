from .planner_wrapper import get_planner_wrapper, prost2ssipp_state
from .planner_wrapper import setup as setup_planner_wrappers
from .ssipp_interface import PlannerExtensions

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

def read_prost_states(file):
	states = []
	with open(file, "r") as f:
		for line in f:
			row = line.split(":")
			states.append(row[1])
	return states

def write_heuristic_values(file, dataset: dict, instance_parser):
	# Compute heuristic
	print("Computing heuristics for " + problem + "...", flush=True)
	# Write results
	with open(file, "w") as f:
		for s, values in dataset.items():
			heuristics = [",".join([name] + list(map(str, h))) for name, h in zip(instance_parser.heuristic_names, values)]
			f.write(":".join([s] + heuristics) + "\n")

