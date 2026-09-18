import sys, os, random
#import faulthandler
#faulthandler.enable()
import numpy as np

from gymnasium import Env
from gymnasium.utils import seeding
from .instance_parser import InstanceParser
from .simulator import RDDLSimulator

class RDDLEnv(Env):
	def __init__(self, domain="navigation", instance="1", **kwargs):
		self.domain = domain + '_mdp'
		self.problem = domain + '_inst_mdp__' + instance
		self.instance = instance

		print("Creating env " + instance + "...")

		# Instance Graph
		self.instance_parser = InstanceParser(domain, instance, **kwargs)

		# Seed Random number generator
		#self.seed()

		self.rddlsim = RDDLSimulator(self.instance_parser)
		self.rddlsim.reset()

		print("Created env " + self.problem)
		
	# Do not understand this yet. Almost all other sample environments have it, so we have it too.
	#def seed(self, seed=None):
		#self.np_random, seed = seeding.np_random(seed)
		#return [seed]

	# Take a real step in the environment. Current state changes.
	def step(self, action_var: int):
		state, reward, done = self.rddlsim.step(action_var)
		return state, reward, done, {}

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
		state = self.rddlsim.reset(state)
		return state, {}

	def random_reset(self):
		return self.reset_to_state(self.random_state())

	def reset(self, seed=None, options=None):
		return self.reset_to_state(None)

	def close(self):
		print("RDDL SIM CLOSED")
		self.rddlsim.close()
		self.rddlsim = None

	def compute_expected_step(self, state, action):
		# Make next state and call get_processed input to get next
		next_state = np.array(state, dtype=np.float32)
		actions = [0] * self.instance_parser.num_actions
		actions[action] = 1
		reward = self.instance_parser.reward_formula(state, actions)
		for (i, node) in enumerate(state):
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
