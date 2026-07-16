import sys, os, random
#import faulthandler
#faulthandler.enable()
import numpy as np
import gym

from gym import Env
from gym.utils import seeding
from gym.envs.rddl.instance_parser import InstanceParser
from gym.envs.rddl.simulator import RDDLSimulator, PyRDDLSimulator

redirect = True
def redirect_stdout():
	sys.stdout.flush() # <--- important when redirecting to files
	newstdout = os.dup(1)
	devnull = os.open(os.devnull, os.O_WRONLY)
	os.dup2(devnull, 1)
	os.close(devnull)
	sys.stdout = os.fdopen(newstdout, 'w')

class RDDLEnv(Env):
	def __init__(self, domain="navigation", instance="1"):
		self.domain = domain + '_mdp'
		self.problem = domain + '_inst_mdp__' + instance
		self.instance = instance

		# Instance Graph
		self.instance_parser = InstanceParser(domain, instance)

		# Seed Random number generator
		self._seed()

		self.done = False  # end_of_episode flag

		if redirect:
			sys.stdout.flush()
			_origstdout = sys.stdout
			_oldstdout_fno = os.dup(sys.stdout.fileno())
			redirect_stdout()
		self.rddlsim = RDDLSimulator(self.instance_parser)
		self.rddlsim.reset()
		if redirect:
			sys.stdout = _origstdout
			sys.stdout.flush()
			os.dup2(_oldstdout_fno, 1)
		
		print("Created env: " + instance)
		
	# Do not understand this yet. Almost all other sample environments have it, so we have it too.
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	# Take a real step in the environment. Current state changes.
	def _step(self, action_var):
		self.state, reward, self.done = self.rddlsim.step(action_var)
		return self.state, reward, self.done, {}

	def test_step(self, action_var):
		self.state, reward, self.done = self.rddlsim.step(action_var)
		# # Handle episode end in case of navigation
		# # Not able to check if robot's position is same as goal state
		# if self.domain == "navigation_mdp" and not(np.any(self.state)):
		#	 self.done = True
		return self.state, reward, self.done, {}

	# Using Sampling
	# UNUSED
	def get_expected_step(self, state, action_var, num_samples=50):
		s = np.array(state)
		self.rddlsim.reset(s)
		expected_rew = 0.0
		next_state_l = []
		for i in range(num_samples):
			next_state, reward = self.rddlsim.step(action_var)
			next_state_l.append(next_state)
			expected_rew += reward
		next_state = np.mean(np.stack(next_state_l), axis=0)
		return next_state, expected_rew/num_samples

	def _reset(self):
		self.state = self.rddlsim.reset()
		self.tstep = 1
		self.done = False
		return self.state, self.done

	def initialize_using_state(self, state):
		self.state = self.rddlsim.reset(state)
		self.tstep = 1
		self.done = False
		return self.state, self.done

	def random_reset(self):
		return self.initialize_using_state(self.random_state())

	def random_state(self):
		if self.domain == 'academic_advising_mdp':
			# (taken, passed)
			num_courses = self.get_num_nodes()
			valid_state_vars_values = [(0,0), (1,0), (1,1)]
			state_temp = np.zeros(self.instance_parser.num_state_vars)
			for i in range(num_courses):
				val = random.choice(valid_state_vars_values)
				state_temp[i] = val[0]
				state_temp[num_courses+i] = val[1]
			return state_temp
		else:
			raise Exception("Random initial state not defined for %s domain"%self.domain)
		# return np.random.choice(2, self.instance_parser.num_state_vars)

	def _set_state(self, state):
		self.state = state
		self.rddlsim.reset(state)

	def _close(self):
		pass

	def get_fluent_features(self, state):
		return np.array(self.instance_parser.get_fluent_features(state))

	def get_numeric_unary_nf_features(self, state):
		return np.array(self.instance_parser.get_numeric_unary_nf_features(state))

	def get_action_details(self):
		return self.instance_parser.get_action_details()

	def get_extended_action_details(self):
		return self.instance_parser.get_extended_action_details()

	def get_nf_features(self):
		return np.array(self.instance_parser.get_nf_features())

	def get_numeric_para_unary_nf_features(self):
		return np.array(self.instance_parser.get_numeric_para_unary_nf_features())

	def get_adjacency_list(self):
		return self.instance_parser.get_adjacency_list()

	def get_extended_adjacency_list(self):
		return self.instance_parser.get_extended_adjacency_list()

	def get_num_adjacency_list(self):
		return self.instance_parser.get_num_adjacency_list()

	def get_feature_dims(self):
		return self.instance_parser.get_feature_dims()

	def get_num_action_nodes(self):
		return self.instance_parser.get_num_action_nodes()

	def get_num_actions(self):
		return self.instance_parser.get_num_actions()

	def get_action_templates(self):
		return self.instance_parser.get_action_templates()

	def get_graph_fluent_features(self, state):
		return np.array(self.instance_parser.get_graph_fluent_features(state))

	def get_num_nodes(self):
		return self.instance_parser.get_num_nodes()

	def get_num_graph_fluents(self):
		return self.instance_parser.get_num_graph_fluents()

	def get_num_type_actions(self):
		return self.instance_parser.get_num_type_actions()

	def get_graph_type(self):
		return self.instance_parser.graph_type

	def get_state_repr(self, state):
		return self.instance_parser.get_state_repr(state)

	def get_action_repr(self, action_probs, action):
		return self.instance_parser.get_action_repr(action_probs, action)

	def get_embedding_repr(self, embeddings):
		return self.instance_parser.get_embedding_repr(embeddings)

	def get_attr(self, attr):  # For getting any attribute in the instance parser class directly
		return self.instance_parser.get_attr(attr)
	
	def get_expected_next_state(self, state, action):
		return self.instance_parser.get_expected_next_state(state, action)

	def get_expected_step_cpt(self, state, action):
		return self.instance_parser.get_expected_next_state_cpt(state, action)

	def get_transition_prob(self, state, action, next_state):
		return self.instance_parser.get_transition_prob(state, action, next_state)

	def get_node_dict(self):
		return self.instance_parser.get_node_dict()

	def get_num_to_action(self):
		return self.instance_parser.get_num_to_action()

	def get_max_reward(self):
		return self.instance_parser.get_max_reward()

	def get_min_reward(self):
		return self.instance_parser.get_min_reward()

if __name__ == '__main__':
	env_name = "RDDL-{}{}-v1".format(sys.argv[1], sys.argv[2])
	env = gym.make(env_name)
	env.detailed_init()
	env.seed(0)
	NUM_EPISODES = 1
	for i in range(NUM_EPISODES):
		reward = 0  # epsiode reward
		rwd = 0  # step reward
		curr, done = env.reset()  # current state and end-of-episode flag
		while not done:
			action = random.randint(0, env.instance_parser.get_num_actions())  # choose a random action
			nxt, rwd, done, _ = env.step(action)  # next state and step reward
			print(('state: {}  action: {}  reward: {} next: {}'.format(curr, action, rwd, nxt)))
			curr = nxt
			reward += rwd
		print(('Episode Reward: {}'.format(reward)))
		print()

	env.close()
