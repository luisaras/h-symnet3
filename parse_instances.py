import sys, os, shutil, argparse, subprocess

def parse_arguments():
	formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
		prog, max_help_position=38
	)
	parser = argparse.ArgumentParser(
		description="Generate dot and pre-parsed instance files for SymNet. "
		"Run this ONLY after running ./install.sh.\n"
		"Run this WITHIN symnet3-env podman environment.\n",
		formatter_class=formatter,
	)
	parser.add_argument("domains", help="domain name(s)",
		nargs="*")
	parser.add_argument("-i", "--ins", help="first and last instances", 
		nargs=2, type=int, default=[1, 254])
	parser.add_argument("--skip", help="skip existing files",
		action="store_true")
	args = parser.parse_args()
	return args

def copy_files_into(src_paths, dst_path):
	with open(os.path.abspath(dst_path), "w") as dst:
		for src_path in src_paths:
			dst.write("\n")
			with open(os.path.abspath(src_path), "r") as src:
				shutil.copyfileobj(src, dst)


def generate_dot(instance, rddl, out_file):
	cwd = os.environ.get('RDDLSIM_ROOT', os.getcwd())
	cmd = ["./run", "rddl.viz.RDDL2Graph", rddl, instance]
	dbn_file = cwd + "/tmp_rddl_graphviz.dot"
	result = None
	try:
		result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
		#print(result.stdout)
	except subprocess.CalledProcessError as e:
		result = e
	except Exception as e:
		print(e)
		sys.exit(1)
	finally:
		if os.path.exists(dbn_file):
			shutil.move(dbn_file, out_file)
			print(f"{instance}.dot generated.")
			return True
		elif result.returncode == 0:
			print(f"PROBLEM generating DBN {instance}.dot!")
			print(result.stdout)
		else:
			print(f"ERROR generating DBN {instance}.dot!")
			print(result.stderr)
			return False

def parse_rddl(instance, rddls, temp_folder, out_file):
	print(f"Starting rddl-parser for {instance}...")
	cmd = ["./prost/rddl-parser"] + rddls + [temp_folder]
	try:
		result = subprocess.run(cmd, capture_output=True, text=True)
		if os.path.exists(f"{temp_folder}/{instance}"):
			shutil.move(f"{temp_folder}/{instance}", out_file)
		else:
			print(F"PROBLEM parsing {instance}!")
			print(e.stdout)
	except subprocess.CalledProcessError as e:
		print(F"ERROR parsing {instance}!")
		print(e.stderr)


def parse_instance(domain, i, skip=False):
	pwd = os.environ.get('PWD', os.getcwd())
	instance=f"{domain}_inst_mdp__{i}"
	domain_folder=f"{pwd}/benchmarks/{domain}"
	domain_rddl=f"{domain_folder}/rddl/{domain}_mdp.rddl"
	instance_rddl=f"{domain_folder}/rddl/{instance}.rddl"
	instance_dbn=f"{domain_folder}/dbn/{instance}.dot"
	instance_parsed=f"{domain_folder}/parsed/{instance}"
	temp_folder=f"{pwd}/temp_{instance}"
	if not skip or not os.path.exists(instance_dbn):
		os.makedirs(temp_folder, exist_ok=True)
		os.makedirs(domain_folder + "/dbn", exist_ok=True)
		print(f"Generating .dot file for {instance}...")
		temp_rddl = temp_folder + "/temp.rddl"
		copy_files_into([instance_rddl, domain_rddl], temp_rddl)
		generate_dot(instance, temp_rddl, instance_dbn)
	if not skip or not os.path.exists(instance_parsed):
		os.makedirs(temp_folder, exist_ok=True)
		os.makedirs(domain_folder + "/parsed", exist_ok=True)
		parse_rddl(instance, [domain_rddl, instance_rddl], temp_folder, instance_parsed)
	if os.path.exists(temp_folder):
		shutil.rmtree(temp_folder)

def parse_all(domains, instances, skip=False):
	for d in domains:
		for i in instances:
			parse_instance(d, i, skip)

if __name__ == "__main__":
	args = parse_arguments()
	parse_all(args.domains, range(args.ins[0], args.ins[1]+1), args.skip)
