import sys, subprocess, os, argparse
from rddl_server import RDDLServer

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
			process = subprocess.run(cmd)
			return self.get_results(process.stdout)

	def run_batch(self, domain_name, instances, save_output=True):
		with self.rddl_server:
			rewards = []
			times = []
			for i in instances: #range(1, 10)
				instance = f"{domain_name}_inst_mdp__{i}"
				cmd = ["python3", self.root + "/prost.py", instance, "[Prost -s 1 -se [IPC2014]]"]
				print(cmd)
				try:
					process = subprocess.run(cmd)
				except subprocess.CalledProcessError as e:
					print(f"Command failed with exit code {e.returncode}")
					print(e.stderr)
					return
				if save_output:
					with open(f'{i}.result', 'w') as f:
						f.write(process.stdout)
				print(process.stdout)
				reward, time = self.get_results(process.stdout)
				rewards.append(result)
				times.append(time)
			return rewards, times

	def get_results(self, output):
		result = output[-500:-1].splitlines()
		for line in result:
			if "AVERAGE REWARD" in line:
				reward = float(result.split()[-1])
			if "PROST complete running time" in line:
				time = float(result.split()[-1])
		return reward, time


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
	parser.add_argument("-r", "--rounds",
		action="store",
		default=30,
		type=int,
		help="Number of episodes/rounds.",
	)
	args = parser.parse_args()
	return args


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
	server = RDDLServer(prost_root, args.domain, args.rounds)
	prost = PROST(server, prost_root)
	print(prost.run(args.instance))
