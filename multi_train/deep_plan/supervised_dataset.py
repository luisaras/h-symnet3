import os
import numpy as np
import pandas as pd
import random
import my_config
import copy
import helper
import tensorflow as tf
from tqdm import tqdm

class SupervisedDataset():
	def __init__(self, instance_list, envs, batch_size):
		self.instance_list = [int(i) for i in instance_list]
		self.envs = envs
		self.batch_size = batch_size

		self.dataset = {}
		print("Loading datasets...")
		for i in tqdm(range(len(self.instance_list))):
			self.load_dataset(i, self.instance_list[i])

	def __iter__(self):
		instance_order = self.instance_list.copy()
		random.shuffle(instance_order)
		for ins in instance_order:
			# Get samples from the current instance's dataset
			env_index, states, actions = self.dataset[ins]
			yield ins, env_index, states, actions

	def load_dataset(self, instance_index, instance):
		domain = self.envs[instance_index].get_domain_name()
		f = os.path.join(my_config.trajectory_dataset_folder, domain, f"{instance}.csv")
		df = pd.read_csv(f, delimiter=":", header=None)

		states = []
		for s in df[1]: # Second column
			state = np.array(s.split(","), dtype="float32")
			states.append(state)

		states = np.stack(states)

		actions = np.expand_dims(np.array(df[2].apply(lambda x: self.envs[instance_index].get_action_num(x)), dtype="float32"), axis=-1)
		rewards = np.expand_dims(np.array(df[3], dtype="float32"), -1)

		if my_config.last_in_dataset:
			# Choose last max_transitions_per_instance or the first. 
			# Choosing the last leads to better results because PROST 
			# learns while it executes.
			# For the paper results we used the first. 
			states = states[-my_config.max_transitions_per_instance:]
			actions = actions[-my_config.max_transitions_per_instance:]
			rewards = rewards[-my_config.max_transitions_per_instance:]
		else:
			states = states[:my_config.max_transitions_per_instance]
			actions = actions[:my_config.max_transitions_per_instance]
			rewards = rewards[:my_config.max_transitions_per_instance]

		new_states, new_actions = [], []
		for i, s in enumerate(states):
			done = False
			for j, s_ in enumerate(new_states):
				if np.array_equal(s, s_):
					done = True
					break
			
			if done:
				continue 

			actions_freq = {}
			for j, a in enumerate(actions):
				if np.array_equal(states[j], s):
					actions_freq[a[0]] = actions_freq.get(a[0], 0) + 1
			
			if 0 in actions_freq and len(actions_freq) == 1:
				new_states.append(states[i])
				new_actions.append([0])
			else:
				if 0 in actions_freq:
					del actions_freq[0]
				most_common_action = None
				for ac in actions_freq:
					if actions_freq[ac] > actions_freq.get(most_common_action, 0):
						most_common_action = ac

				new_states.append(states[i])
				new_actions.append([most_common_action])
				
		new_states, new_actions = np.array(new_states), np.array(new_actions)
		states, actions = copy.deepcopy(new_states), copy.deepcopy(new_actions)

		combined_dataset = np.hstack([states, actions])
		np.random.shuffle(combined_dataset)
		states = combined_dataset[:, :-1]
		states = [tuple(states[i]) for i in range(states.shape[0])]
		actions = combined_dataset[:, -1]

		# one hot actions
		actions = np.eye(self.envs[instance_index].get_num_actions())[actions.astype(np.int32)]
		self.dataset[instance] = (instance_index, states, tf.convert_to_tensor(actions))
