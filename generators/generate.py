#!/bin/python3
# This file is for generating new domains using the generators

import subprocess, os, sys, random, shutil, argparse
import config

def parse_arguments():
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=38
    )
    parser = argparse.ArgumentParser(
        description="Generate domain instances.",
        formatter_class=formatter,
    )
    parser.add_argument("domain", help="domain name")
    parser.add_argument("instance", type=int, help="number of first instance")
    parser.add_argument("-n", "--num_instances", type=int, default=1, help="number of instances to generate")
    parser.add_argument("-d", "--dataset", default="train", help="train, test, val or something custom")
    parser.add_argument("-s", "--seed", type=int, default=None, help='random seed; if -1, set seed to the instance number')
    parser.add_argument("-v", "--verbose", help="log errors", action="store_true")
    parser.add_argument("--skip", help="skip existent files", action="store_true")
    args = parser.parse_args()
    return args

class Generator():
    def __init__(self, rddlsim_folder):
        self.script_args = config.script_args
        self.script_name = config.script_name
        
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.rddlsim_folder = rddlsim_folder
        self.cp = '.commons-math3-3.6.1.jar'
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
    
    def args_string(self, domain, dataset, args):
        arg_str = []
        for i, arg_type in enumerate(self.script_args[domain + "-" + dataset]):
            arg_str.append(f'{arg_type["param_name"]}: {args[i]}')
        return ','.join(arg_str)

    def validate_args(self, domain, args):
        # CONSTRAINTS
        if domain == 'academic_advising_prob':
            # FOR TRAINING
            num_courses_per_level_idx = 3
            num_levels_idx = 2
            num_courses = int(args[num_courses_per_level_idx]) * int(args[num_levels_idx])
            if num_courses >= 20:
                print(f"Courses too high({num_courses})! Retrying...")
                return False
            else:
                return True
        return True

    def get_args(self, domain, dataset, output_dir, instance_name):
        args = []
        for arg_types in self.script_args[domain + "-" + dataset]:
            if arg_types['param_name'] == 'output-dir':
                args.append(output_dir)
            elif arg_types['param_name'] == 'instance-name':
                args.append(instance_name)
            elif arg_types['type'] == int:
                args.append(str(random.randint(arg_types['min'], arg_types['max'])))
            elif arg_types['type'] == float:
                args.append(str(random.uniform(arg_types['min'], arg_types['max'])))
            elif arg_types['type'] == str:
                args.append(arg_types['value'])
        return args

    def generate_instance(self, domain, dataset, instance, verbose=False):
        output_dir = self.get_output_dir(domain)
        script_name = self.script_name[domain]
        instance_name = f'{domain}_inst_mdp__{instance}'
        if ".py" in script_name:
            command = ['python3', script_name]
            directory = os.path.abspath(os.path.join(self.folder, "domains"))
        else:
            command = ['java', '-cp', self.cp, script_name]
            directory = os.path.abspath(self.rddlsim_folder)

        args = self.get_args(domain, dataset, output_dir, instance_name)
        while not self.validate_args(domain, args):
            args = self.get_args(domain)

        if self.seed:
            args.append(str(self.seed))
        
        out, err = subprocess.DEVNULL, subprocess.DEVNULL
        if verbose:
            out, err = None, None
        
        print(f"Generating RDDL with arguments [{self.args_string(domain, dataset, args)}]")
        try:
            subprocess.run(command + args, cwd=directory, stdout=out, stderr=err)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(e.stderr)

    def get_output_dir(self, domain):
        return os.path.abspath(os.path.join(self.folder, "..", "benchmarks", domain, "rddl"))

    def check_instance_file(self, domain, instance):
        output_dir = self.get_output_dir(domain)
        instance_name = f'{domain}_inst_mdp__{instance}'
        file = os.path.join(output_dir, instance_name + ".rddl")
        if os.path.exists(file):
            print("Instance already generated: " + instance_name)
            return True
        return False


if __name__ == '__main__':
    try:
        rddlsim_root = os.environ["RDDLSIM_ROOT"]
    except KeyError:
        err_msg = (
            "Error: an environment variable RDDLSIM_ROOT pointing to "
            "your RDDLSIM_ROOT installation must be setup."
        )
        print(err_msg)
        sys.exit()
    args = parse_arguments()
    generator = Generator(rddlsim_root)
    if args.seed and args.seed >= 0:
        generator.set_seed(args.seed)
    for i in range(args.instance, args.instance + args.num_instances):
        if args.seed == -1:
            generator.set_seed(i)
        instance = str(i)
        if args.skip and generator.check_instance_file(args.domain, instance):
            continue
        generator.generate_instance(args.domain, args.dataset, instance, args.verbose)
