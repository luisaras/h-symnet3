import os
import numpy as np
import time
import math

from model_factory import ModelFactory
import helper

class PolicyMonitor(object):
	def __init__(self, envs, network, domain, summary_writer, model_factory):
		self.domain = domain
		self.envs = envs
		self.summary_writer = summary_writer
		self.model_factory = model_factory
		self.network = network
		if model_factory is None:
			self.network_copy = None
		else:
			self.network_copy = model_factory.create_network()
		self.eval_step = 1
		self.attn_maps = []
		self.node_embs = []

	def copy_params(self):
		ModelFactory.copy_params(self.network_copy.trainable_variables, self.network.trainable_variables)

	def predict_action(self, env, state, get_attn_map, get_node_emb):
		import tensorflow as tf
		if get_attn_map:
			logits, attn = self.network_copy.policy_prediction([state], env, training=False,
				return_attn_coef=True)
			self.attn_maps.append((state, attn.numpy()))
		elif get_node_emb:
			logits, node_emb = self.network_copy.policy_prediction([state], env, training=False,
				return_attn_coef=False, return_node_emb=True)
			self.node_embs.append((state, node_emb))
		else:
			logits = self.network_copy.policy_prediction([state], env,training=False)
		return tf.argmax(tf.reshape(logits, [-1])).numpy()

	def simulate(self, policy, num_episodes=5, graph_file=None, cache_actions=True, verbose=False):
		from tqdm import tqdm

		start_time = time.time()
		mean_times = []
		mean_crewards = []
		mean_lengths = []
		std_crewards = []
		std_error_crewards = []
		crewards = []
		lengths = []
		times = []

		image_name = None
		if graph_file is not None:
			graph_file = os.path.abspath(os.path.join(graph_file, "graphs"))

		for i, env in enumerate(self.envs):
			if verbose:
				print("env = %d" % (i))
			env_crewards = []
			env_lengths = []
			env_times = []

			#  Update file name for plotting
			if graph_file is not None:
				dir_name = os.path.abspath(os.path.join(graph_file, str(env.get_instance_num())))
				os.makedirs(dir_name, exist_ok=True)

			for j in tqdm(range(num_episodes), desc=f'Simulating Env {env.get_instance_num()}'):
				state_action_cache = {}  # Caches actions so that a forward pass is not done each time
				initial_state, done = env.reset()
				state = initial_state
				episode_reward = 0.0
				length = 0
				episode_start = time.time()
				print("----------------------------\n\n") if verbose else None
				while not done:
					#  Update file name for plotting
					if graph_file is not None:
						image_name = os.path.abspath(os.path.join(dir_name, f"{length}.png"))
						# TODO?
					if state in state_action_cache:
						action = state_action_cache[state]
					else:
						action = policy(env, state)
						if cache_actions:
							state_action_cache[state] = action
					# Env step
					next_state, reward, done, _ = env.step(action)
					if verbose:
						print("Episode: ", length)
						state_str = env.get_state_str(state)
						print("State: " + state_str)
						action_str = env.get_action_str(action)
						print("Action taken:", action_str)
						print("Reward:", reward)
						print("--------------------------------------------------------------\n\n")
					episode_reward += reward
					length += 1
					state = next_state
				# Store this episode's stats
				env_crewards.append(episode_reward)
				env_lengths.append(length)
				env_times.append(time.time() - episode_start)
			# Store this env's states
			crewards.append(env_crewards)
			lengths.append(env_lengths)
			times.append(env_times)
			# Average/std over episodes
			mean_creward = np.mean(env_crewards)
			mean_crewards.append(mean_creward)
			mean_length = np.mean(env_lengths)
			mean_lengths.append(mean_length)
			mean_time = np.mean(env_times)
			mean_times.append(mean_time)
			std_creward = np.std(env_crewards)
			std_crewards.append(std_creward)
			std_error_crewards.append(std_creward / math.sqrt(num_episodes))
			print("Instance:", i, "Mean reward:", mean_creward)

		total_time = time.time() - start_time

		print("\n==============")
		print("std_crewards = " + str(std_error_crewards))
		print("mean_crewards = " + str(mean_crewards))
		print("==============")

		results = dict(
			total_time=total_time,
			# Per instance
			creward_means=mean_crewards,
			creward_std_errors=std_error_crewards,
			creward_stds=std_crewards,
			length_means=mean_lengths,
			time_means=mean_times,
			# Per instance, per episode
			ep_crewards=crewards,
			ep_lengths=lengths,
			ep_times=times,
		)
		return results


	def eval_policy(self, num_episodes=5, graph_file=None,
			log_file=None, get_attn_map=False, get_node_emb=False, verbose=False, cache_actions=True):
		import tensorflow as tf

		if self.network_copy is None:
			policy = lambda env, state: env.get_random_action()
		else:
			policy = lambda env, state: self.predict_action(env, state, get_attn_map, get_node_emb)

		self.attn_maps.clear()
		self.node_embs.clear()
		results = self.simulate(policy, num_episodes, graph_file, cache_actions and not get_attn_map, verbose)

		if get_attn_map:
			results["attn_maps"] = self.attn_maps
		if get_node_emb:
			results["node_embs"] = self.node_embs
		return results
