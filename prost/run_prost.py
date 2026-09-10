#!/bin/python3
import sys, subprocess, os, argparse, shutil, copy
import logging, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from rddl_server import RDDLServer

VERBOSE = False

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Run PROST from directory defined "
		"by environment variable PROST_ROOT.",
		formatter_class=formatter,
	)
	parser.add_argument("domains", help="domain name(s)",
		nargs="+")
	parser.add_argument("-i", "--instances", help="first and last instances", 
		nargs=2, type=int, default=[1, 254])
	parser.add_argument("-w", "--workers", help="number of workers",
		type=int,
		default=1)
	parser.add_argument("-r", "--rounds", help="number of episodes/rounds",
		action="store",
		type=int,
		default=30)
	parser.add_argument("-d", "--dir_pattern", help="directory pattern with rddl files",
		default=os.path.join("{prost_folder}", "testbed", "bechmarks", "{domain}"))
	parser.add_argument("-l", "--log", help="log output file or directory.",
		default=None)
	parser.add_argument("-p", "--port", help="shift applied to the RDDLSim port",
		type=int,
		default=0)
	parser.add_argument("-f", "--prost_folder", help="PROST root folder",
		default=None)
	parser.add_argument("-v", "--verbose", help="print log on screen",
		action="store_true")
	parser.add_argument("--skip", help="skip instance if dataset already exists",
		action="store_true")
	args = parser.parse_args()
	if args.prost_folder is None:
		try:
			args.prost_folder = os.environ["PROST_ROOT"]
		except KeyError:
			err_msg = (
				"Error: an environment variable PROST_ROOT pointing to "
				"your PROST installation must be setup."
			)
			print(err_msg)
			sys.exit()
	args.cwd = None
	return args


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


def check_files(domain, instances, log): # True if they all exist
	log = os.path.join(log.format(domain=domain))
	for i in instances:
		file = os.path.join(log, f"{i}.result")
		if not os.path.exists(file):
			return False
	return True


def set_domain(args, d):
	args.domain = d
	args.domain_folder = args.dir_pattern.format(domain=d, prost_folder=args.prost_folder)
	os.makedirs(args.log.format(domain=args.domain), exist_ok=True)


def run_new_server(args, cwd=None):
	if args.skip and check_files(args.domain, args.instances):
		return
	print(f"Running server in {args.domain_folder}, port {args.port}")
	server = RDDLServer(args.prost_folder, args.domain_folder, args.rounds, args.port)
	prost = PROST(args.prost_folder, args.port, cwd, args.verbose)
	with server:
		if len(args.instances) > 1 or args.workers > 1:
			# Run num_instances instances
			print(prost.run_batch(args.domain, args.instances, args.log, args.skip))
		else:
			# Run single instance of given name
			print(prost.run(args.instances[0], args.log))


def run_worker(args_i):
	# Create temp folder for instance
	instance = f'{args_i.domain}_inst_mdp__{args_i.instances[0]}'
	domain = f'{args_i.domain}_mdp'
	folder = "temp_" + instance
	os.makedirs(folder, exist_ok=True)
	shutil.copy(os.path.join(args_i.domain_folder, instance + ".rddl"), folder)
	shutil.copy(os.path.join(args_i.domain_folder, domain + ".rddl"), folder)
	args_i.domain_folder = folder
	args_i.port += args_i.instances[0]
	print("Planning for instance " + instance)
	run_new_server(args_i, folder)
	shutil.rmtree(folder)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(processName)s - %(levelname)s - %(message)s"
)

def init_worker():
    # This runs once when each worker process starts
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

if __name__ == "__main__":
	args = parse_arguments()
	args.instances = range(args.instances[0], args.instances[1]+1)
	if args.workers == 1 and (len(args.domains) > 1 or len(args.instances) > 1): # Sequential batch
		print("Running PROST sequentially...")
		for d in args.domains:
			set_domain(args, d)
			run_new_server(args)
	else:
		print(f"Running PROST with {args.workers} async workers...")
		worker_args = []
		for d in args.domains:
			set_domain(args, d)
			for i in args.instances:
				args_i = copy.copy(args)
				args_i.instances = [i]
				worker_args.append(args_i)
		if len(worker_args) > 1:
			with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
				futures = [executor.submit(run_worker, arg) for arg in worker_args]
				for i, future in enumerate(as_completed(futures)):
					try:
						future.result()
					except Exception as exc:
						logging.error(f"Task {i} generated an exception: {exc}")
						traceback.print_exc()
						sys.exit(1)
		else: # Only one instance
			run_worker(worker_args[0])

