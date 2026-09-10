import sys, os, ctypes, re, tempfile, shutil
import numpy as np
import threading
from pyRDDLGym import RDDLEnv

curr_dir_path = os.path.dirname(os.path.realpath(__file__))

silent_parsing = True
_origstdout = sys.stdout
_oldstdout_fno = os.dup(sys.stdout.fileno())
stdout_lock = threading.Lock()

def redirect_stdout():
	# Redirect
	sys.stdout.flush() # <--- important when redirecting to files
	newstdout = os.dup(1)
	devnull = os.open(os.devnull, os.O_WRONLY)
	os.dup2(devnull, 1)
	os.close(devnull)
	sys.stdout = os.fdopen(newstdout, 'w')

def revert_stdout():
	sys.stdout = _origstdout
	sys.stdout.flush()
	os.dup2(_oldstdout_fno, 1)
		

class RDDLSimulator:

	def __init__(self, instance_parser):
		self._tstep = 1  # current time step
		self._state = None
		self.horizon = instance_parser.horizon
		self.initial_state = tuple(instance_parser.initial_state_values)
		if "termination" in instance_parser.state_to_num:
			self.termination_id = instance_parser.state_to_num["termination"]
			print("Termination var: ", self.termination_id)
		else:
			self.termination_id = -1

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

		if silent_parsing:
			with stdout_lock:
				redirect_stdout()
				self._load_instance(instance_parser.parsed_instance_file)
				revert_stdout()
		else:
			self._load_instance(instance_parser.parsed_instance_file)

	def _load_instance(file):
		# Better without the explicit encoding
		parsed_instance_file_byteobject = file.encode()
		parsed_instance_file_ctype = ctypes.create_string_buffer(parsed_instance_file_byteobject, len(parsed_instance_file_byteobject))
		try:
			self.simlib.parse(parsed_instance_file_ctype.value)
		except Exception as e:
			print("Error with the rddl sim library: ")
			print(e)

	def _get_next_state(self, s: tuple, a: int) -> tuple[tuple, float]:
		# Convert state and action to c-types
		array = (ctypes.c_double * len(s))(*s) # C array
		action = (ctypes.c_int)(a)
		# Call Simulator
		reward = self.simlib.step(array, len(s), action)
		#state = np.array(array, dtype=np.int8)
		return tuple(array), reward

	def _is_done(self, s):
		if self._tstep > self.horizon:
			return True 
		elif self.termination_id >= 0:
			return s[self.termination_id] == 1
		return False

	def reset(self, state: tuple = None) -> tuple[float, ...]:
		self._tstep = 1  # current time step
		if state is None:
			state = self.initial_state
		self._state = state
		return state

	def lookahead(self, state: tuple, action_var: int) -> tuple[tuple, float, bool]:
		next_state, reward = self._get_next_state(state, action_var)
		return next_state, reward, self._is_done(next_state)

	def step(self, action_var: int) -> tuple[tuple, float, bool]:
		# Convert state and action to c-types
		s = self._state
		array = (ctypes.c_double * len(s))(*s) # C array
		action = (ctypes.c_int)(action_var)

		# Call Simulator
		reward = self.simlib.step(array, len(s), action)
		#state = np.array(array, dtype=np.int8)
		self._state = tuple(array)

		# Advance time step
		self._tstep = self._tstep + 1

		return self._state, reward, self._is_done(self._state)


class PyRDDLSimulator:

	def __init__(self, instance_parser):
		self._tstep = 1 # current time step
		self._state = None
		self._env = RDDLEnv(instance_parser.domain_file, instance_parser.instance_file)
		self.num_vars = instance_parser.num_state_vars
		self.action_names = instance_parser.num_to_action
		self.var_numbers = instance_parser.state_to_num
		self.horizon = instance_parser.horizon
		if "termination" in instance_parser.state_to_num:
			self.termination_id = instance_parser.state_to_num["termination"]
			print("Termination var: ", self.termination_id)
		else:
			self.termination_id = -1

	def _set_state(self, state: tuple):
		# 1. Access the underlying simulator state dictionary
		# Keys are string representations of the grounded fluents
		internal_state = self._env._simulator._state
		# 2. Manually overwrite specific state fluents
		for rddl_str, i in self.instance_parser.state_to_num.items():
			gym_key = rddl_str_to_gym_key(rddl_str)
			internal_state[gym_key] = state[i]

	def reset(self, state: tuple) -> tuple:
		self._tstep = 1
		obs, _ = self._env.reset()
		if state is None:
			self._state = self.gym_obs_to_tuple(obs)
		else:
			self._state = state
			self._set_state(state)
		return self._state

	def _step_wrap(self, action_var: int):
		rddl_str = self.action_names[action_var]
		gym_key = rddl_str_to_gym_key(rddl_str)
		obs, reward, terminated, truncated, _ = self._env.step(gym_key)
		s = self.gym_obs_to_tuple(obs)
		if self._tstep > self.horizon:
			truncated = True 
		if self.termination_id >= 0 and s[self.termination_id] == 1:
			terminated = True
		return s, reward, terminated or truncated

	def lookahead(self, state: tuple, action_var: int):
		self._set_state(state)
		s, r, done = self._step_wrap(action_var)
		self._set_state(self._state)
		return s, r, done

	def step(self, action_var):
		self._tstep = self._tstep + 1
		s, r, done = self._step_wrap(action_var)
		self._state = s
		return s, r, done

	def gym_obs_to_tuple(self, obs):
		s = [0] * self.num_vars
		for gym_key, val in obs.items():
			rddl_str = gym_key_to_rddl_str(gym_key)
			i = self.var_numbers[rddl_str]
			s[i] = val
		return tuple(s)


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

