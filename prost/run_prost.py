#!/bin/python3
import sys, subprocess, os, argparse
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
	parser.add_argument("domain", help="Domain name.")
	parser.add_argument("instance", help="Instance name.")
	parser.add_argument("-r", "--rounds", help="Number of episodes/rounds.",
		action="store",
		default=30,
		type=int
	)
	parser.add_argument("-d", "--directory", help="Directory with rddl files.",
		default=None)
	args = parser.parse_args()
	return args


class PROST:

	def __init__(self, rddl_server, prost_root="../../prost"):
		self.root = prost_root
		self.rddl_server = rddl_server

	def run(self, instance):
		with self.rddl_server:
			# Run PROST here
			#inst = f"{args.domain}_inst_mdp__{args.instance}"
			cmd = ["python3", self.root + "/prost.py", instance, "[Prost -s 1 -se [IPC2014]]"]
			print(cmd)
			try:
				process = subprocess.Popen(cmd,
					stdout=subprocess.PIPE,    # Capture stdout
					#stderr=subprocess.STDOUT,  # Redirect stderr into stdout so everything is in one place
					text=True,                 # Decode bytes to a string
					#check=True                 # Raise error on crash
				)
				output = ""
				for line in process.stdout:
					print(line, end="")        # Show on screen instantly
					output += "\n" + line  # Save to list
				process.wait()
				return self.get_results(output)
			except subprocess.CalledProcessError as e:
				print(f"Command failed with exit code {e.returncode}")
				print(e.stdout)
				return None

	def run_batch(self, domain_name, instances, save_output=True):
		with self.rddl_server:
			rewards = []
			times = []
			for i in instances: #range(1, 10)
				instance = f"{domain_name}_inst_mdp__{i}"
				cmd = ["python3", self.root + "/prost.py", instance, "[Prost -s 1 -se [IPC2014]]"]
				print(cmd)
				try:
					process = subprocess.Popen(cmd,
						stdout=subprocess.PIPE,    # Capture stdout
						#stderr=subprocess.STDOUT,  # Redirect stderr into stdout so everything is in one place
						text=True,                 # Decode bytes to a string
						#check=True                 # Raise error on crash
					)
					output = ""
					for line in process.stdout:
						print(line, end="")        # Show on screen instantly
						output += "\n" + line  # Save to list
					process.wait()
				except subprocess.CalledProcessError as e:
					print(f"Command failed with exit code {e.returncode}")
					print(e.stdout)
					return
				if save_output:
					with open(f'{i}.result', 'w') as f:
						f.write(output)
				reward, time = self.get_results(output)
				rewards.append(result)
				times.append(time)
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
	server = RDDLServer(prost_root, "" if args.directory else args.domain, args.directory, args.rounds)
	prost = PROST(server, prost_root)
	print(prost.run(args.instance))
