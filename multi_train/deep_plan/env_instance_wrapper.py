import os, sys, json
import numpy as np
import my_config
import networkx as nx
import re

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, "..", ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path

from heuristics import compute_heuristics
from gym import Wrapper

class EnvInstanceWrapper(Wrapper):
	def __init__(self, env):
		super(EnvInstanceWrapper, self).__init__(env)
		# Nonfluents
		self.nf_features = env.get_nf_features()
		if my_config.repeat_graph_nf:
			# Vishal End: Add non-fluent values again due to adding new nodes.
			graph_nf = env.get_graph_nf_features()
			graph_nf = np.broadcast_to(graph_nf[np.newaxis, :],
				(self.nf_features.shape[0], graph_nf.shape[0]))
			# Add unparameterized nonfluents to all nodes
			self.nf_features = np.hstack((self.nf_features, graph_nf))
		else:
			self.graph_nf_features = env.get_graph_nf_features()
		# Action data
		self.extended_action_details = env.get_extended_action_details()
		self.num_action_nodes = env.get_num_action_nodes()
		# Adjacency matrices
		self.adjacency_mat = env.get_adjacency_mats()

		self._cache_distances = None
		self._cache_masks = None

		if my_config.heuristics:
			compute_heuristics.wrapper_type = my_config.init_heuristics
			domain_folder = env.instance_parser.domain_folder
			ppddl_file = os.path.join(domain_folder, 'ppddl', env.problem + ".ppddl")
			self.planner_wrapper = compute_heuristics.get_planner_wrapper(ppddl_file, env.problem, my_config.heuristics)
			self.planner_wrapper.normalization = my_config.heuristic_normalization
			self.planner_wrapper.instance_parser = env.instance_parser
			h_file = os.path.join(my_config.heuristics_dataset_folder,
				env.instance_parser.domain, 
				env.instance_parser.instance + '.csv')
			self.planner_wrapper.add_cache(h_file)

    #def get_node_feature_dim(self):
		#state, _ = self.envs[0].reset()
		#_, node_features, graph_features, _ = self.get_parsed_state([state], 0)
		#return node_features.shape[-1]

	def get_distance_mask(self):
		if self._cache_masks is None:
			n = len(self.env.instance_parser.node_dict)
			m = len(self.env.instance_parser.fluent_nodes)
			mask = np.eye(n)
			mask += np.pad(np.ones((m, m)), ((0, n-m), (0, n-m)))
			mask = np.clip(mask, 0, 1)
			self._cache_masks = mask
		return self._cache_masks

	def get_distance_mat(self, adjacency_mat, dbn_distance=True):
		batch_size = adjacency_mat.shape[0]
		if self._cache_distances is None:
			# Compute shortest distance on the first adjacency
			graph = nx.from_dict_of_lists(self.env.instance_parser.adjacency_lists[0], create_using=nx.DiGraph)
			distance_dict = dict(nx.all_pairs_shortest_path_length(graph))
			num_nodes = self.env.get_num_nodes()
			distance_mat = np.zeros((1, num_nodes, num_nodes))
			for i in range(distance_mat.shape[1]):
				for j in range(distance_mat.shape[2]):
					distance_mat[0][i][j] = distance_dict[i].get(j, -1)
			distance_mat = distance_mat/max(np.max(distance_mat), 1)
			self._cache_distances = distance_mat
		distance_mat = np.stack([self._cache_distances for _ in range(batch_size)])
		return distance_mat

	def get_processed_adj_mat(self, batch_size):
		return np.array([[self.adjacency_mat[i] for batch in range(batch_size)] for i in range(len(self.adjacency_mat))]).astype(np.float32)

	def get_processed_input(self, states):
		def state2feature(state):
			feature_arr = np.array(self.env.instance_parser.get_fluent_features(state))
			feature_arr = np.hstack((feature_arr, self.nf_features))
			if my_config.use_type_encoding:
				feature_arr = np.hstack((feature_arr, self.env.instance_parser.type_encoding))
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def get_processed_graph_input(self, states):
		def state2feature(state):
			feature_arr = self.env.instance_parser.get_graph_fluent_features(state)
			if not my_config.repeat_graph_nf:
				feature_arr.extend(self.graph_nf_features)
			if my_config.heuristics:
				h = self.planner_wrapper.get_heuristics(state)
				feature_arr.extend(h)
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def estimate_successor_heuristics(self, states, action_var):
		return np.array([
			self.planner_wrapper.get_heuristics(self.env.sample_step(s, action_var)[0]) 
				for s in states
		])

	def get_random_action(self):
		return np.random.randint(0, self.env.get_num_actions())

	def get_random_node():
		return np.random.randint(0, len(self.env.instance_parser.node_dict)-1)

	def get_random_fluent_node():
		return np.random.randint(0, len(self.env.instance_parser.fluent_nodes)-1)

	def get_action_details(self):
		return self.env.instance_parser.detailed_action

	def get_extended_action_details(self):
		return self.env.instance_parser.extended_action_details

	def get_action_affects(self):
		return self.env.instance_parser.action_affects

	def get_attr(self, batch_size, name): # For getting any attribute in the instance parser class directly
		return [self.env.instance_parser.get_attr(name) for _ in range(batch_size)]

	def get_domain_name(self):
		return self.env.instance_parser.domain

	def get_num_actions(self):
		return self.env.instance_parser.num_actions

	def get_num_action_types(self):
		return self.env.instance_parser.num_types_action

	def get_num_adjacency_list(self):
		return len(self.env.instance_parser.adjacency_lists)

	def get_num_edge_types(self):
		return len(self.env.instance_parser.dbn_edge_types_to_idx)

	def get_action_num(self, name):
		if name == 'noop()':
			return 0
		else:
			return self.env.instance_parser.action_to_num[name]

	# Debug stuff

	def __str__(self):
		return f"EnvInstanceWrapper({self.env.problem})"

	def get_instance_num(self):
		return self.env.instance

	def get_state_str(self, state):
		state_repr = self.envs[i].env.instance_parser.get_state_repr(state)
		if type(state_repr) == dict:
			return json.dumps(state_repr, sort_keys=True, indent=4)
		else:
			return state_repr

	def get_action_str(self, action_var):
		if action == 0:
			return "noop"
		else:
			return self.env.get_num_to_action()[action_var]