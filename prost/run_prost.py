#!/usr/bin/env python3
# =============================================================================
import sys, os, argparse, copy
import logging, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(curr_dir_path, ".."))
if root_path not in sys.path:
	sys.path = [root_path] + sys.path

from prost import run_new_server, run_new_server_single

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

def set_domain(args, d):
	args.domain = d
	args.domain_folder = args.dir_pattern.format(domain=d, prost_folder=args.prost_folder)
	os.makedirs(args.log.format(domain=args.domain), exist_ok=True)


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
				futures = [executor.submit(run_new_server_single, arg) for arg in worker_args]
				for i, future in enumerate(as_completed(futures)):
					try:
						future.result()
					except Exception as exc:
						logging.error(f"Task {i} generated an exception: {exc}")
						traceback.print_exc()
						sys.exit(1)
		else: # Only one instance
			run_new_server_single(worker_args[0])

