import os, sys, json
import numpy as np
import tensorflow as tf
import networkx as nx

from gymnasium import Wrapper

class EnvInstanceWrapper(Wrapper):
	def __init__(self, env, planner_wrapper=None, repeat_graph_nf=True, use_type_encoding=True, heuristic_samples=1, prune_actions=False):
		super(EnvInstanceWrapper, self).__init__(env)
		self.use_type_encoding = use_type_encoding
		self.heuristic_samples = heuristic_samples
		self.prune_actions = prune_actions
		# Nonfluents
		nf_features = tf.convert_to_tensor(env.instance_parser.nf_features, dtype=tf.double) # 2D array
		graph_nf = tf.convert_to_tensor(env.instance_parser.unpara_nf_values, dtype=tf.double) # 1D array
		if repeat_graph_nf:
			num_samples = tf.shape(nf_features)[0]
			# Add non-fluent values again due to adding new nodes.
			graph_nf_repeated = tf.tile(tf.expand_dims(graph_nf, axis=0), [num_samples, 1])
			# Add unparameterized nonfluents to all nodes
			self.nf_features = tf.concat([nf_features, graph_nf_repeated], axis=1)
			self.graph_nf_features = None
		else:
			self.nf_features = nf_features
			self.graph_nf_features = env.instance_parser.unpara_nf_valuess
		# Action data
		self.num_action_nodes = env.get_num_action_nodes()
		# Adjacency matrices
		self.adjacency_mats = []
		for adj_list in env.instance_parser.adjacency_lists: # One list per layer
			num_nodes = len(adj_list)
			edges = []
			for node, neighbors in adj_list.items():
				for dst_node in neighbors:
					edges.append([dst_node, node]) # Transposed
			edges = tf.constant(edges, dtype=tf.int64)
			n_edges = tf.shape(edges)[0]
			matrix_shape = tf.constant([num_nodes, num_nodes], dtype=tf.int64)
			values = tf.ones([n_edges], dtype=tf.double) # values should be tf.float32 for sparse
			adj_mat = tf.scatter_nd(edges, values, matrix_shape)
			#adj_mat = tf.SparseTensor(indices=edges, values=values, dense_shape=matrix_shape)
			# (Optional) Reorder the sparse tensor to ensure it is optimized for computations
			#adj_mat = tf.sparse.reorder(adj_mat)
			self.adjacency_mats.append(adj_mat)
		# Type features
		if use_type_encoding:
			self.type_features = tf.convert_to_tensor(env.instance_parser.type_encoding, dtype=tf.double)
		# For heuristics
		self.planner_wrapper = planner_wrapper

		self._cache_distances = None
		self._cache_masks = None

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
			self._cache_masks = tf.convert_to_tensor(mask)
		return self._cache_masks

	def get_distance_mat(self, batch_size, dbn_distance=True) -> tf.Tensor:
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
			self._cache_distances = tf.convert_to_tensor(distance_mat, dtype=tf.double)
		#distance_mat = tf.stack([self._cache_distances for _ in range(batch_size)])
		distance_mat = tf.tile(tf.expand_dims(self._cache_distances, axis=0), [batch_size, 1, 1, 1])
		return distance_mat

	def get_processed_adj_mat(self, batch_size) -> tf.Tensor:
		def repeat_adj_mat(adj_mat):
			return tf.tile(tf.expand_dims(adj_mat, axis=0), [batch_size, 1, 1])
		return tf.stack(list(map(repeat_adj_mat, self.adjacency_mats)))
		# return np.array([[self.adjacency_mat[i] for batch in range(batch_size)] for i in range(len(self.adjacency_mat))]).astype(np.float32)

	def get_processed_input(self, states) -> tf.Tensor:
		def state2tensor(state):
			features = self.env.instance_parser.get_fluent_features(state)
			tensors = [tf.convert_to_tensor(features, dtype=tf.double), self.nf_features]
			if self.type_features is not None:
				tensors.append(self.type_features)
			return tf.concat(tensors, axis=1)
		return tf.stack(list(map(state2tensor, states)))

	def get_processed_graph_input(self, states) -> tf.Tensor:
		def state2tensor(state):
			features = self.env.instance_parser.get_graph_fluent_features(state)
			if self.graph_nf_features is not None:
				features.extend(self.graph_nf_features)
			if self.planner_wrapper is not None:
				h = self.planner_wrapper.get_heuristics(state)
				features.extend(h)
			return tf.convert_to_tensor(features, dtype=tf.double)
		return tf.stack(list(map(state2tensor, states)))

	def get_successor_heuristic_input(self, states, action_var) -> tf.Tensor:
		def state2samples(state): # n_samples X n_heuristics 
			return [self.planner_wrapper.get_heuristics(self.env.sample_step(state, action_var)[0])
						for i in range(self.heuristic_samples)]
		# batch_size X n_samples X n_heuristics 
		features = tf.convert_to_tensor(list(map(state2samples, states)), dtype=tf.double)
		# batch_size X n_heuristics 
		return tf.reduce_mean(features, axis=1)

	def get_random_node():
		return np.random.randint(0, len(self.env.instance_parser.node_dict)-1)

	def get_random_fluent_node():
		return np.random.randint(0, len(self.env.instance_parser.fluent_nodes)-1)

	def get_action_details(self):
		return self.env.instance_parser.detailed_action

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

	def get_feature_dims(self):
		node_dim = self.env.instance_parser.fluent_feature_dims + self.env.instance_parser.nonfluent_feature_dims
		graph_dim = self.env.instance_parser.graph_fluent_feature_dims + 1
		if self.graph_nf_features is None:
			return node_dim + self.env.instance_parser.graph_nonfluent_feature_dims, graph_dim
		else:
			return node_dim, graph_dim + self.env.instance_parser.graph_nonfluent_feature_dims

	# Actions

	def get_action_num(self, name) -> str:
		if name == 'noop()':
			return 0
		else:
			return self.env.instance_parser.action_to_num[name]

	def select_random_action(self, state) -> int:
		if self.prune_actions:
			mask = self.get_prune_mask(state)
			actions = tf.where(mask != 0)
			i = np.random.randint(0, tf.shape(actions)[0])
			return actions[i].numpy()
		else:
			return np.random.randint(0, self.get_num_actions())

	def select_best_action(self, state, scores) -> int:
		if self.prune_actions:
			mask = self.get_prune_mask(state)
			scores -= 10e9 * (1.0 - mask)
		return tf.argmax(scores).numpy()

	def sample_actions(self, states, probs):
		batch_size = tf.shape(probs)[0]
		masks = None
		if self.prune_actions:
			masks = tf.stack([self.get_prune_masks(state) for state in states])
			# Expected shape is (batch_size, num_actions)
			probs = prune_probs(probs, masks, axis=-1)
		return tf.random.categorical(logits=tf.math.log(probs), num_samples=batch_size, dtype=tf.int32)  # Return sampled actions
		        
	def get_prune_mask(self, state):
		# TODO
		return tf.ones([self.get_num_actions()])

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

	def close(self):
		self.env.close()
		if self.planner_wrapper is not None:
			self.planner_wrapper.close()

@tf.function
def prune_probs(probs, mask, axis=0):
	probs *= mask
	total_prob = tf.reduce_sum(probs, axis=axis, keepdims=True)
	safe_total = tf.clip_by_value(total_prob, clip_value_min=1e-9, clip_value_max=tf.double.max)
	return probs / safe_total