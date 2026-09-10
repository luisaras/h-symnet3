import numpy as np
import my_config as my_config
from model_factory import ModelFactory

class Worker(object):
	def __init__(self, worker_id, domain, envs, global_network,
			model_factory, lock, policy_monitor=None, summary_writer=None):
		self.worker_id = worker_id
		self.domain = domain
		self.envs = envs
		self.model_factory = model_factory
		self.global_network = global_network
		self.global_counter = model_factory.global_counter
		self.summary_writer = summary_writer
		self.current_instance = 0
		self.state = None
		self.local_network = model_factory.create_network()
		self.local_network.init_network(self.envs[0])
		self.lock = lock
		self.policy_monitor = policy_monitor

		# ! Only for academic advising
		self.degree_completed = False
		self.steps_taken = 0

		# For reward prediction
		self.action_noop = 0

		if self.policy_monitor is not None:
			self.global_network.init_network(self.envs[0])
			self.policy_monitor.network_copy.init_network(self.envs[0])
			self.policy_monitor.copy_params()
		self.copy_global_params()

		# To train using a dataset
		self.save_transitions = self.train_from_dataset = False

		self.masks = [None, None]
		self.masks = [None, None]

	def copy_global_params(self, acquire_lock=True):
		if not acquire_lock:
			ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
			return
		self.lock.acquire()
		ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
		self.lock.release()

	def eval_policy_net(self, num_episodes=5, graph_file=None, get_attn_map=False, get_node_emb=False):
		self.policy_monitor.copy_params()
		return self.policy_monitor.eval_policy(num_episodes, graph_file, save_model, cache_actions=True, get_attn_map, get_node_emb=get_node_emb, verbose=True)