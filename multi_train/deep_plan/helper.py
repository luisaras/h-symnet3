import os
import numpy as np
import shutil
import sys
import my_config
import symnet3_config
from datetime import datetime

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, "..", ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path
import gym
from gym.envs.rddl import instance_parser

from heuristics import compute_heuristics

def load_config(file=None):
	if file:
		with open(file, "r") as file:
			file_content = file.read()
			exec(file_content, globals(), my_config.__dict__)
	if my_config.mode == 'no_dist': # SymNet2.0
		my_config.add_aux_loss = False
		my_config.decay_aux_loss = False
	elif my_config.mode == "kl": # SymNet3.0+KL
		my_config.add_aux_loss = True
		my_config.decay_aux_loss = False
	elif my_config.mode == "no_kl": # SymNet3.0-KL 
		my_config.add_aux_loss = False
		my_config.decay_aux_loss = False
	elif my_config.mode == "kl_decay": # SymNet3.0+KL_{decay}
		my_config.add_aux_loss = True
		my_config.decay_aux_loss = True
	if my_config.net_config:
		with open(my_config.net_config, "r") as file:
			file_content = file.read()
			exec(file_content, globals(), symnet3_config.__dict__)
	if my_config.mode == "no_dist":
		symnet3_config.se_params["use_preprocess_layer"] = False
		symnet3_config.se_params["num_preprocess"] = 4 # Filter size in each GAT (num of
		symnet3_config.se_params["num_postprocess"] = 4 # Filter size in each GAT (num of
		symnet3_config.se_params["use_distance_mat"] = False
		symnet3_config.se_params["use_preprocess_layer"] = False

def get_instance_names():
	train_instances,test_instances = [],[]
	for instance_num in my_config.train_instance.strip().split(","):
		instance = "{}".format(instance_num)
		train_instances.append(instance)
	for instance_num in my_config.test_instance.strip().split(","):
		instance = "{}".format(instance_num)
		test_instances.append(instance)
	instances = []
	instances.extend(train_instances)
	instances.extend(test_instances)
	return train_instances,len(train_instances),test_instances,len(test_instances),instances

def make_envs(instances):
	instance_parser.setup(my_config)
	compute_heuristics.wrapper_type = my_config.init_heuristics
	envs = []
	for instance in instances:
		try: 
			env_name = "RDDL-{}{}-v1".format(my_config.domain, instance)
			env = gym.make(env_name)
			if my_config.heuristics:
				domain_folder = env.instance_parser.domain_folder
				ppddl_file = os.path.join(domain_folder, 'ppddl', env.problem + ".ppddl")
				planner_wrapper = compute_heuristics.get_planner_wrapper(ppddl_file, env.problem, my_config.heuristics)
				planner_wrapper.normalization = my_config.heuristic_normalization
				env.instance_parser.set_planner_wrapper(planner_wrapper)
			envs.append(env)
		except ValueError as e:
			print(e)
	return envs

def get_env_metadata(envs_):
	num_nodes_list = []
	num_valid_actions_list = []
	num_graph_fluent_list = []
	num_adjacency_list = []
	for env_ in envs_:
		num_nodes_list.append(env_.get_num_nodes())
		num_valid_actions_list.append(env_.get_num_actions())
		num_graph_fluent_list.append(env_.get_num_graph_fluents())
		num_adjacency_list.append(env_.get_num_adjacency_list())
	return num_nodes_list, num_valid_actions_list, num_graph_fluent_list, num_adjacency_list

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

def create_modelfactory_args(**kwargs):
	args = {}
	for key,value in kwargs.items():
		args[key] = value
	if "policynet_optim" not in args.keys():
		args["policynet_optim"] = None
	args["grad_clip_value"] = my_config.grad_clip_value

	return args

def add_network_args(args, env, MODEL_DIR, copy_config=True):
	args["general_params"] = symnet3_config.general_params
	args["se_params"] = symnet3_config.se_params
	args["ad_params"] = symnet3_config.ad_params
	args["ge_params"] = symnet3_config.ge_params
	args["tm_params"] = symnet3_config.tm_params

	args["se_params"]["num_se"] = env.get_num_adjacency_list()
	if my_config.fc_adjacency:
		args["se_params"]["num_se"] += 1
	args["ad_params"]["num_action_templates"] = env.get_num_type_actions()

	if copy_config:
		shutil.copy(os.path.abspath("symnet3_config.py"), MODEL_DIR)
		if my_config.net_config:
			shutil.copy(os.path.abspath(my_config.net_config), MODEL_DIR + "/symnet3_config_mods.py")

def get_adj_mat_from_list(adjacency_list):
	l = len(adjacency_list)
	adj_mat = np.array(np.zeros((l, l), dtype=np.int32), dtype=np.int32)
	for key, value in adjacency_list.items():
		for val in value:
			adj_mat[key, val] = 1
	for i in range(l):
		adj_mat[i][i] = 1
	return adj_mat

def write_content(file_path, content):
	with open(file_path, 'a') as f:
		f.write(content)

def backup_source_code(model_dir, config_file=None):
    py_source = os.path.join(model_dir, "source_")
    shutil.copy(os.path.abspath("my_config.py"), py_source + "my_config.py")
    if config_file:
        shutil.copy(os.path.abspath(config_file), py_source + "config_mods.py")
    shutil.copy(os.path.abspath("policy_monitor.py"), py_source + "policy_monitor.py")
    shutil.copy(os.path.abspath("networks/symnet3/symnet3.py"), py_source + "symnet3.py")

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
		#backup_source_code(model_dir, config_file)
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