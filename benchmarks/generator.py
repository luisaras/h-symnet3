#!/bin/python3
# This file is for generating new domains using the generators

import subprocess, os, sys, random, shutil, argparse, importlib
from benchmarks import config

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

def get_rddlsim_root():
    try:
        return os.environ["RDDLSIM_ROOT"]
    except KeyError:
        err_msg = (
            "Error: an environment variable RDDLSIM_ROOT pointing to "
            "your RDDLSIM_ROOT installation must be setup."
        )
        print(err_msg)
        sys.exit()

class Generator():
    def __init__(self, rddlsim_folder=None, verbose=False):
        self.script_args = config.script_args
        self.verbose = verbose
        
        self.folder = os.path.dirname(os.path.realpath(__file__))
        if rddlsim_folder is None:
            self.rddlsim_folder = get_rddlsim_root()
        else:
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

    def get_args(self, domain, dataset, output_dir, instance_name):
        args = []
        for arg_types in self.script_args[domain + "-" + dataset]:
            if arg_types['param_name'] == 'output-dir':
                args.append(output_dir)
            elif arg_types['param_name'] == 'instance-name':
                args.append(instance_name)
            elif arg_types['type'] == int:
                args.append(random.randint(arg_types['min'], arg_types['max']))
            elif arg_types['type'] == float:
                args.append(random.uniform(arg_types['min'], arg_types['max']))
            elif arg_types['type'] == str:
                args.append(arg_types['value'])
        return args

    def generate_instance(self, domain, dataset, instance):
        output_dir = self.get_output_dir(domain)
        instance_name = f'{domain}_inst_mdp__{instance}'
        gen_module = importlib.import_module("benchmarks." + domain + ".generate")
        args = self.get_args(domain, dataset, output_dir, instance_name)
        while not gen_module.validate_args(dataset, args):
            args = self.get_args(domain, dataset, output_dir, instance_name)
        if self.seed:
            gen_module.rng.seed(self.seed)
            print(self.seed)
        if self.verbose:
            print(f"Generating RDDL with arguments [{self.args_string(domain, dataset, args)}]")
        gen_module.create_instances(*args)

    def get_output_dir(self, domain):
        return os.path.abspath(os.path.join(self.folder, domain, "rddl"))

    def check_instance_file(self, domain, instance):
        output_dir = self.get_output_dir(domain)
        instance_name = f'{domain}_inst_mdp__{instance}'
        file = os.path.join(output_dir, instance_name + ".rddl")
        if os.path.exists(file):
            print("Instance already generated: " + instance_name)
            return True
        return False

    def generate_all(self, domain, dataset, i, n, seed=None, skip=False):
        if seed and seed >= 0:
            self.set_seed(seed)
        for instance in range(i, i + n):
            if seed == -1:
                self.set_seed(instance)
            instance = str(instance)
            if skip and self.check_instance_file(domain, instance):
                continue
            self.generate_instance(domain, dataset, instance)


if __name__ == '__main__':
    args = parse_arguments()
    generator = Generator(rddlsim_root, args.verbose)
    generator.generate_all(args.domain, args.dataset, args.instance, args.num_instances, args.seed, args.skip)
