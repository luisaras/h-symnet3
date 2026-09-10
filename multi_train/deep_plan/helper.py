import os, sys
import numpy as np
import shutil
import my_config
import symnet3_config
from env_instance_wrapper import EnvInstanceWrapper
from datetime import datetime

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, "..", ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path
import gym
from gym.envs.rddl import instance_parser

def load_config_mods(file, module):
	with open(file, "r") as f:
		file_content = f.read()
		exec(file_content, globals(), module.__dict__)

def load_config(file=None):
	if file:
		load_config_mods(file, my_config)
	load_net_config(my_config.net_config)

def load_net_config(net_config=None):
	if net_config:
		load_config_mods(net_config, symnet3_config)
	if not my_config.add_aux_loss: # SymNet2.0 config
		symnet3_config.se_params["use_preprocess_layer"] = False
		symnet3_config.se_params["num_preprocess"] = 4 # Filter size in each GAT
		symnet3_config.se_params["num_postprocess"] = 4 # Filter size in each GAT
		symnet3_config.se_params["use_distance_mat"] = False
		symnet3_config.se_params["use_preprocess_layer"] = False

def get_instance_names():
	train_instances, test_instances = [], []
	for instance_num in my_config.train_instance.strip().split(","):
		instance = "{}".format(instance_num)
		train_instances.append(instance)
	for instance_num in my_config.test_instance.strip().split(","):
		instance = "{}".format(instance_num)
		test_instances.append(instance)
	return train_instances, test_instances

def make_envs(instances):
	instance_parser.setup(my_config)
	envs = []
	for instance in instances:
		try: 
			env_name = "RDDL-{}{}-v1".format(my_config.domain, instance)
			env = gym.make(env_name)
			envs.append(EnvInstanceWrapper(env))
		except ValueError as e:
			print(e)
	return envs

def failsafe():
	print("================================================================")
	print("domain = " + my_config.domain)
	print("model_dir = " + my_config.model_dir)
	print("train_instance = " + my_config.train_instance)
	print("test_instance = " + my_config.test_instance)
	print("t_max = " + str(my_config.t_max))
	print()
	print("symnet3_params = " + str(symnet3_config.ge_params))
	print("================================================================")
	input()


def write_content(file_path, content):
	try:
		with open(file_path, 'a') as f:
			f.write(content)
	except OSError as e:
		print("Error writing file: " + str(file_path), file=sys.stderr)
		print(e)
		sys.exit(e.errno)

def copy_files_into(src_paths, dst_path):
	with open(os.path.abspath(dst_path), "a") as dst:
		for src_path in src_paths:
			dst.write("\n")
			with open(os.path.abspath(src_path), "r") as src:
				shutil.copyfileobj(src, dst)

def backup_settings(model_dir, config_file=None):
	py_source = os.path.join(model_dir, "source_")
	shutil.copy(os.path.abspath("my_config.py"), py_source + "my_config.py")
	if config_file:
		copy_files_into([config_file], py_source + "my_config.py")
	#shutil.copy(os.path.abspath("policy_monitor.py"), py_source + "policy_monitor.py")
	#shutil.copy(os.path.abspath("networks/symnet3/symnet3.py"), py_source + "symnet3.py")
	shutil.copy(os.path.abspath("symnet3_config.py"), py_source + "symnet3_config.py")
	if my_config.net_config:
		copy_files_into([my_config.net_config], py_source + "symnet3_config.py")

def restore_settings(model_dir):
	py_source = os.path.join(model_dir, "source_")
	config = py_source + "my_config.py"
	if os.path.exists(config):
		load_config(config)
	net_config = py_source + "symnet3_config.py"
	if os.path.exists(net_config):
		load_net_config(net_config)

def get_model_dir(config_file, create=False):
	model_name = f"{my_config.domain}_{my_config.exp_description}"
	model_dir = os.path.abspath(os.path.join(my_config.model_dir, model_name))
	checkpoint_dir = os.path.join(model_dir, "checkpoints")
	#train_summary_path = os.path.join(model_dir, "train_summaries")
	#val_summary_path = os.path.join(model_dir, "val_summaries")
	log_file = os.path.join(model_dir, "meta_logging.csv")
	if create:
		os.makedirs(model_dir, exist_ok=True)
		os.makedirs(checkpoint_dir, exist_ok=True)
		if my_config.restore_config:
			backup_settings(model_dir, config_file)
		#os.makedirs(os.path.join(model_dir, "instances"), exist_ok=True)
		log_header = "init: " + str(datetime.now()) + "\n"
		log_header += my_config.test_instance + "\n"
		write_content(log_file, log_header)
	else:
		if not os.path.exists(checkpoint_dir):
			print("Incorrect checkpoint_dir :\n" + checkpoint_dir)
			exit(-1)
	return model_dir, checkpoint_dir, log_file

def read_checkpoint_log(log_file):
	with open(log_file, 'r') as f:
		best_ckpt, best_rew = 1, -1000000
		lines = f.readlines()
		i = len(lines) - 1
		if i < 2:
			return [] # No checkpoints
		while i > 0 and not lines[i].startswith("init"):
			i -= 1
		return lines[i+2:]