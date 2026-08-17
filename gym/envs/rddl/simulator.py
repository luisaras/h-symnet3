import sys, os, ctypes, re, tempfile, shutil
import numpy as np
from pyRDDLGym import RDDLEnv

curr_dir_path = os.path.dirname(os.path.realpath(__file__))

class RDDLSimulator:

	def __init__(self, instance_parser):
		self.tstep = 1  # current time step
		self.horizon = instance_parser.horizon
		self.initial_state = instance_parser.initial_state_values

		lib_path = os.path.join(curr_dir_path, "clibxx.so")
		if not os.path.isfile(lib_path):
			print("Lib file not found: " + lib_path)
			sys.exit(-1)
		lib_copy_path = tempfile.NamedTemporaryFile().name
		shutil.copy2(lib_path, lib_copy_path)
		if not os.path.isfile(lib_copy_path):
			print("Failed to copy lib file to: " + lib_copy_path)
			sys.exit(-1)
		print("Copied rddlsim library: " + lib_path)

		self.simlib = ctypes.CDLL(lib_copy_path)
		print("Loaded rddlsim library.")

		self.simlib.step.restype = ctypes.c_double

		# Better without the explicit encoding
		parsed_instance_file_byteobject = instance_parser.parsed_instance_file.encode()
		parsed_instance_file_ctype = ctypes.create_string_buffer(parsed_instance_file_byteobject, len(parsed_instance_file_byteobject))

		try:
			self.simlib.parse(parsed_instance_file_ctype.value)
		except Exception as e:
			print("Error with the rddl sim library: ")
			print(e)

	# State is np.array
	def reset(self, state=None):
		if not state:
			self.tstep = 1  # current time step
			state = np.array(self.initial_state)
		self.state = state.tolist()
		return state

	def step(self, action_var):
		# Convert state and action to c-types
		s = self.state # as list
		array = (ctypes.c_double * len(s))(*s) # C array
		action = (ctypes.c_int)(action_var)

		# Call Simulator
		reward = self.simlib.step(array, len(s), action)
		state = np.array(array, dtype=np.int8)
		self.state = state.tolist()

		# Advance time step
		done = False
		self.tstep = self.tstep + 1
		if self.tstep > self.horizon:
			done = True
		return state, reward, done


class PyRDDLSimulator:

	def __init__(self, instance_parser):
		self.instance_parser = instance_parser
		self.env = RDDLEnv(instance_parser.domain_file, instance_parser.instance_file)

	def reset(self, state=None):
		obs, info = self.env.reset()
		if state:
			# 1. Access the underlying simulator state dictionary
			# Keys are string representations of the grounded fluents
			internal_state = self.env._simulator._state
			# 2. Manually overwrite specific state fluents
			for rddl_str, i in self.instance_parser.state_to_num.items():
				gym_key = rddl_str_to_gym_key(rddl_str)
				internal_state[gym_key] = state[i]
			obs = self.env._simulator.get_state()
		return self.gym_obs_to_array(obs)

	def step(self, action_var):
		rddl_str = self.instance_parser.num_to_action[action_var]
		gym_key = rddl_str_to_gym_key(rddl_str)
		obs, reward, terminated, truncated, info = self.env.step(gym_key)
		return self.gym_obs_to_array(obs), reward, terminated or truncated

	def gym_obs_to_array(self, obs):
		s = np.zeros(self.instance_parser.num_state_vars)
		for gym_key, val in obs.items():
			rddl_str = gym_key_to_rddl_str(gym_key)
			i = self.instance_parser.state_to_num[rddl_str]
			s[i] = val
		return s

def rddl_str_to_gym_key(rddl_string: str) -> str:
	# Remove all whitespace from the string
	clean_str = rddl_string.replace(" ", "")
	
	# Check if the action has parameters (contains parentheses)
	match = re.match(r"^([\w-]+)\((.+)\)$", clean_str)
	if not match:
		# It's a scalar action with no arguments (e.g., "push")
		return clean_str
		
	fluent_name, args_block = match.groups()
	
	# Split individual objects by commas
	objects = args_block.split(",")
	
	# Combine using pyRDDLGym's specific underscore syntax
	return f"{fluent_name}___{'__'.join(objects)}"

# --- Tests ---
#print(rddl_str_to_gym_key("push"))				 # Output: push
#print(rddl_str_to_gym_key("velocity(car1)"))		# Output: velocity___car1
#print(rddl_str_to_gym_key("move(obj1, obj2)"))	  # Output: move___obj1__obj2
#print(rddl_str_to_gym_key("load(robot1, box3, a)")) # Output: load___robot1__box3__a

def gym_key_to_rddl_str(gym_key: str) -> str:
    # Check if the key contains the triple underscore separator
    if "___" not in gym_key:
        # It's a scalar state fluent with no parameters
        return gym_key
        
    # Split the base fluent name from the parameters block
    fluent_name, args_block = gym_key.split("___", 1)
    
    # Split individual objects using the double underscore separator
    objects = args_block.split("__")
    
    # Reconstruct the text format without any spaces
    return f"{fluent_name}({','.join(objects)})"

# --- Tests ---
#print(gym_key_to_rddl_str("temperature"))                 # Output: temperature
#print(gym_key_to_rddl_str("pos___car1"))                  # Output: pos(car1)
#print(gym_key_to_rddl_str("link___nodeA__nodeB"))         # Output: link(nodeA,nodeB)
#print(gym_key_to_rddl_str("grid___x1__y2__z3"))           # Output: grid(x1,y2,z3)

