import sys, os, random
#import faulthandler
#faulthandler.enable()
import numpy as np
import gym

from gym import Env
from gym.utils import seeding
from gym.envs.rddl.instance_parser import InstanceParser
from gym.envs.rddl.simulator import RDDLSimulator, PyRDDLSimulator

class RDDLEnv(Env):
	def __init__(self, domain="navigation", instance="1"):
		self.domain = domain + '_mdp'
		self.problem = domain + '_inst_mdp__' + instance
		self.instance = instance

		print("Creating env " + instance + "...")

		# Instance Graph
		self.instance_parser = InstanceParser(domain, instance)

		# Seed Random number generator
		self._seed()

		self.done = False  # end_of_episode flag
		self.state = None # tuple

		self.rddlsim = RDDLSimulator(self.instance_parser)
		self.rddlsim.reset()

		print("Created env " + self.problem)
		
	# Do not understand this yet. Almost all other sample environments have it, so we have it too.
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	# Take a real step in the environment. Current state changes.
	def _step(self, action_var: int):
		self.state, reward, done = self.rddlsim.step(action_var)
		return self.state, reward, done, {}

	# Using Sampling
	def sample_step(self, state: tuple, action_var: int):
		next_state, reward, done = self.rddlsim.lookahead(state, action_var)
		return next_state, reward, done, {}

	# UNUSED
	def mean_step(self, state: tuple, action_var: int, num_samples=50):
		expected_rew = 0.0
		next_state_l = []
		done_chance = 0
		for i in range(num_samples):
			next_state, reward, done = self.sample_step(state, action_var)
			if done: done_chance += 1
			next_state_l.append(np.array(next_state))
			expected_rew += reward
		next_state = np.mean(np.stack(next_state_l), axis=0)
		return next_state, expected_rew/num_samples, done_chance/num_samples

	def reset_to_state(self, state):
		self.state = self.rddlsim.reset(state)
		return self.state, {}

	def random_reset(self):
		return self.reset_to_state(self.random_state())

	def _reset(self):
		return self.reset_to_state(None)

	def _close(self):
		pass

	def compute_expected_step(self, state, action):
		# Make next state and call get_processed input to get next
		next_state = np.array(state, dtype=np.float32)
		actions = [0] * self.instance_parser.num_actions
		actions[action] = 1
		reward = self.instance_parser.reward_formula(state, actions)
		for (i, node) in enumerate(state):
			# next_state[i] = self.eval_formula(self.formulae[i], state, actions)
			next_state[i] = self.instance_parser.formulae[i](state, actions)
		return next_state, reward

	def compute_transition_prob(self, state, action, next_state):
		prob = 1.0
		bernoulli_probs, _ = self.compute_expected_step(state, action)
		for i, state_var in enumerate(next_state):
			if state_var == 1:
				prob *= (bernoulli_probs[i])
			else:
				prob *= (1 - bernoulli_probs[i])
		return prob

	def get_extended_action_details(self):
		return np.array(self.instance_parser.extended_detailed_action)

	def get_nf_features(self):
		# Get features (one for each node) of node constants (non-fluent)
		return np.array(self.instance_parser.nf_features)

	def get_graph_nf_features(self):
		# Get features of global graph constants (non-fluent)
		return np.array(self.instance_parser.unpara_nf_values)

	def get_adjacency_mats(self):
		# Convert adj lists to adj matrices
		adj_mats = []
		for adj_list in self.instance_parser.adjacency_lists: # One list per layer
			num = len(adj_list)
			adj_mat = np.array(np.zeros((num, num), dtype=np.int32), dtype=np.int32)
			for i in range(num):
				adj_mat[i][i] = 1
			for node, neighbors in adj_list.items():
				for n in neighbors:
					adj_mat[node, n] = 1
			adj_mats.append(adj_mat)
		return adj_mats

	def get_feature_dims(self):
		node_dim = self.instance_parser.fluent_feature_dims + self.instance_parser.nonfluent_feature_dims
		graph_dim = self.instance_parser.graph_fluent_feature_dims
		return node_dim + self.instance_parser.graph_nonfluent_feature_dims, graph_dim

	def get_num_action_nodes(self):  # Number of nodes corresponding to state variable tuples
		return len(self.instance_parser.extended_node_dict) - len(self.instance_parser.node_dict)

	def get_action_templates(self):
		return self.instance_parser.action_template_to_num.keys()

	def get_num_nodes(self):
		return self.instance_parser.num_nodes

	def get_num_graph_fluents(self):
		return len(self.instance_parser.unpara_state_names)

	def get_graph_type(self):
		return self.instance_parser.graph_type

	def get_num_to_action(self):
		return self.instance_parser.num_to_action
