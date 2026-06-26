#!/bin/python3
import sys, subprocess, os, argparse, shutil
from rddl_server import RDDLServer

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Run PROST from directory defined "
		"by environment variable PROST_ROOT.",
		formatter_class=formatter,
	)
	parser.add_argument("domain", help="domain name")
	parser.add_argument("instance", help="name or number of first instance")
	parser.add_argument("-n", "--num_instances", help="number of instances (if batch)",
		type=int,
		default=None)
	parser.add_argument("-r", "--rounds", help="number of episodes/rounds",
		action="store",
		type=int,
		default=30)
	parser.add_argument("-d", "--directory", help="directory with rddl files",
		default=None)
	parser.add_argument("-l", "--log", help="log output file or directory (if batch).",
		default=None)
	parser.add_argument("-p", "--port", help="RDDL sim port",
		type=int,
		default=0)
	args = parser.parse_args()
	return args


class PROST:

	def __init__(self, prost_root="../../prost", port_shift=0, cwd='.'):
		self.root = prost_root
		self.port = str(2323 + port_shift)
		self.cwd = cwd

	def run(self, instance_name, log_file=None):
		cmd = ["python3", self.root + "/prost.py", instance_name, "-p", self.port, "[Prost -s 1 -se [IPC2014]]"]
		print(cmd)
		try:
			process = subprocess.Popen(cmd,
				stdout=subprocess.PIPE,    # Capture stdout
				#stderr=subprocess.STDOUT,  # Redirect stderr into stdout so everything is in one place
				text=True,                 # Decode bytes to a string
				#check=True                 # Raise error on crash
				cwd=self.cwd,
			)
			output = ""
			for line in process.stdout:
				print(line, end="")    # Show on screen instantly
				output += line  # Save to list
			process.wait()
			if log_file:
				with open(log_file, 'w') as f:
					f.write(output)
			return self.get_results(output)
		except subprocess.CalledProcessError as e:
			print(f"Command failed with exit code {e.returncode}")
			print(e.stdout)
			return None

	def run_batch(self, domain_name, instances, log_folder=None):
		rewards = {}
		times = {}
		for i in instances:
			if log_folder:
				log_file = f'{log_folder}/{i}.result'
			else:
				log_file = None
			instance = f"{domain_name}_inst_mdp__{i}"
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
			print("Error trying to parsing output.")
			return None, None

if __name__ == "__main__":
	# Check if the environment variable PROST_ROOT exists
	try:
		prost_root = os.environ["PROST_ROOT"]
	except KeyError:
		err_msg = (
			"Error: an environment variable PROST_ROOT pointing to "
			"your PROST installation must be setup."
		)
		print(err_msg)
		sys.exit()

	args = parse_arguments()

	if args.directory:
		# Custom rddl folder
		domain_folder = args.directory
	if not domain_folder:
		# {domain} folder within prost testbed benchmark
		domain_folder = os.path.join(prost_root, "testbed", "bechmarks", args.domain)

	cwd = None
	if args.num_instances == 1:
		# Single instance: create temp folder
		instance = f'{args.domain}_inst_mdp__{args.instance}'
		domain = f'{args.domain}_mdp'
		cwd = "temp_" + instance
		os.makedirs(cwd, exist_ok=True)
		shutil.copy(os.path.join(domain_folder, instance + ".rddl"), cwd)
		shutil.copy(os.path.join(domain_folder, domain + ".rddl"), cwd)
		domain_folder = cwd

	server = RDDLServer(prost_root, domain_folder, args.rounds, args.port)
	prost = PROST(prost_root, args.port, cwd)
	with server:
		if args.num_instances:
			# Run num_instances instances
			args.instance = int(args.instance)
			instances = range(args.instance, args.instance + args.num_instances)
			print(prost.run_batch(args.domain, instances, args.log))
		else:
			# Run single instance of given name
			print(prost.run(args.instance, args.log))

	if cwd:
		shutil.rmtree(cwd)