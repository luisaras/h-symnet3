import os, subprocess

class PROST:

	def __init__(self, prost_root="../../prost", port_shift=0, cwd='.', verbose=False):
		self.root = prost_root
		self.port = str(2323 + port_shift)
		self.cwd = cwd
		self.verbose = verbose

	def run(self, instance_name, log_file=None):
		cmd = ["python3", self.root + "/prost.py", instance_name, "-p", self.port, "[Prost -s 1 -se [IPC2014]]"]
		try:
			process = subprocess.Popen(cmd,
				stdout=subprocess.PIPE,    # Capture stdout
				#stderr=subprocess.STDOUT, # Redirect stderr into stdout so everything is in one place
				text=True,                 # Decode bytes to a string
				#check=True                # Raise error on crash
				cwd=self.cwd,
			)
			output = ""
			for line in process.stdout:
				if self.verbose: print(line, end="")    # Show on screen instantly
				output += line  # Save to list
			process.wait()
			if log_file:
				with open(log_file, 'w') as f:
					f.write(output)
				print("Salved PROST log: " + log_file)
			return self.get_results(output)
		except subprocess.CalledProcessError as e:
			print(f"Command failed with exit code {e.returncode}")
			print(e.stdout)
			return None

	def run_batch(self, domain_name, instances, log_folder=None, skip=False):
		# Run several instances, sequentially, in the same server
		rewards = {}
		times = {}
		for i in instances:
			instance = f"{domain_name}_inst_mdp__{i}"
			if log_folder:
				log_file = os.path.join(log_folder.format(domain=domain_name), f'{i}.result')
			else:
				log_file = None
			if skip and os.path.exists(log_file):
				print("Skipped instance " + instance)
				continue
			reward, time = self.run(instance, log_file)
			rewards[i] = reward
			times[i] = time
		return rewards, times

	def get_results(self, output):
		result = output[-500:-1].splitlines()
		reward, time = None, None
		for line in result:
			if "AVERAGE REWARD" in line:
				reward = float(line.split()[-1])
			if "PROST complete running time" in line:
				time = float(line.split()[-1])
		if reward and time:
			return reward, time
		else:
			print("Error trying to parse output.")
			return None, None
