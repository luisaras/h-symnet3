import os, sys
from itertools import count
import my_config
import symnet3_config

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
network_path = os.path.abspath(os.path.join(curr_dir_path, "networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path

symnet3_args = {"general_params", "se_params", "ad_params", "ge_params", "tm_params"}


def get_args(env_wrapper, **args):
	if "policynet_optim" not in args.keys():
		args["policynet_optim"] = None
	args["grad_clip_value"] = my_config.grad_clip_value
	# Networks args
	args["general_params"] = symnet3_config.general_params
	args["se_params"] = symnet3_config.se_params
	args["ad_params"] = symnet3_config.ad_params
	args["ge_params"] = symnet3_config.ge_params
	args["tm_params"] = symnet3_config.tm_params
	args["se_params"]["num_se"] = env_wrapper.get_num_adjacency_list()
	args["ad_params"]["num_action_templates"] = env_wrapper.get_num_action_types()
	args["general_params"]["num_heuristics"] = len(my_config.heuristics)
	args["general_params"]["remove_dbn"] = my_config.remove_dbn
	if my_config.split_dbn:
		args["se_params"]["num_edge_types"] = env_wrapper.get_num_edge_types()
	return args


class ModelFactory:
	def __init__(self, env_wrapper=None, **kwargs):
		self.args = get_args(env_wrapper, **kwargs)
		self.global_counter = count()
		self.policynet_optim = self.args["policynet_optim"]
		self.grad_clip_value = self.args["grad_clip_value"]

		self.ckpt = None
		self.ckpt_manager = None

		self.total_num_updates = 0.0
		self.total_examples = 0.0

		self.total_episodes = 0.0
		self.num_complete = 0.0
		self.avg_steps = 0.0

	def set_ckpt_metadata(self, ckpt, ckpt_manager):
		self.ckpt = ckpt
		self.ckpt_manager = ckpt_manager

	def create_network(self): #	 Creates a combined network
		import symnet3.symnet3 as symnet3
		return symnet3.SymNet3(**dict((k, self.args[k]) for k in symnet3_args))

	@staticmethod
	def copy_params(target_variables, source_variables):
		for t, s in zip(target_variables, source_variables):
			t.assign(s)  # copies the variables of global model (g) into local model (l)

	def load_ckpt(self, ckpt_num=None):
		if self.ckpt_manager.latest_checkpoint:
			if ckpt_num:
				ckpt_path = f'{self.ckpt_manager._checkpoint_prefix}-{ckpt_num}'
				print(("Loading model checkpoint: {}".format(ckpt_path)))
				self.ckpt.restore(ckpt_path)
			else:
				print(("Loading model checkpoint: {}".format(self.ckpt_manager.latest_checkpoint)))
				self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
		else:
			print("Training new model")

	def save_ckpt(self):
		ckpt_loc = self.ckpt_manager.save()
		print("Model saved at: " + ckpt_loc)
		return ckpt_loc

	def get_ckpt_num(self):
		ckpt = self.ckpt_manager.latest_checkpoint
		return ckpt.replace(self.ckpt_manager._checkpoint_prefix + "-", "")


