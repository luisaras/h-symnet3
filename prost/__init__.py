import os, shutil
from .rddl_server import RDDLServer
from .prost_server import PROST

def check_files(domain, instances, log): # True if they all exist
	log = os.path.join(log.format(domain=domain))
	for i in instances:
		file = os.path.join(log, f"{i}.result")
		if not os.path.exists(file):
			return False
	return True

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

def run_new_server_single(args_i):
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
