import argparse
import sys
import os
import my_config
import csv
import pandas as pd
import numpy as np
import logging, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curr_dir_path,"..",".."))
if parent_dir_path not in sys.path:
    sys.path = [parent_dir_path] + sys.path

import helper
from policy_monitor import PolicyMonitor
from model_factory import ModelFactory

def evaluate_instance(model_dir, ckpt_dir, test_instance, num_episodes, process_index):
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use GPU for inference
    val_summary_writer = None

    test_envs = helper.make_envs([test_instance])
    print("Envs created.")

    policynet_optim = tf.keras.optimizers.RMSprop(learning_rate=my_config.lr, rho=0.99, momentum=0.0, epsilon=1e-6)
    model_factory = ModelFactory(test_envs[0], policynet_optim=policynet_optim)

    network = model_factory.create_network()
    network.init_network(test_envs[0])
    policy_monitor = PolicyMonitor(
        envs=test_envs,
        network=network,
        domain=my_config.domain,
        summary_writer=val_summary_writer,
        model_factory=model_factory)
    print("Created policy monitor.")

    # Create CheckpointManager
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)

    # Load weights
    model_factory.load_ckpt(my_config.exact_checkpoint)

    print("In process:", process_index)
    policy_monitor.copy_params()
    results = policy_monitor.eval_policy(num_episodes=num_episodes)

    test_envs[0].close()

    sys.stdout = sys.__stdout__
    print("Rewards:", results["ep_crewards"])

    return results


def test(model_dir, ckpt_dir):
    _, test_instances = helper.get_instance_names()
    num_episodes = my_config.num_testing_episodes

    results_all_instances = {}
    with ThreadPoolExecutor(max_workers=my_config.num_threads) as executor:
        arg_list = [(model_dir, ckpt_dir, instance, num_episodes, i) for i, instance in enumerate(test_instances)]
        futures = [executor.submit(evaluate_instance, *arg) for arg in arg_list]
        for future, arg in zip(as_completed(futures), arg_list):
            try:
                results_all_instances[arg[2]] = future.result()
            except Exception as exc:
                print(f"Task {arg[4]} generated an exception: {exc}", file=sys.stderr)
                traceback.print_exc()
                sys.exit(1)

    csv_file = os.path.abspath(os.path.join(model_dir, "results.csv"))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Instance", "Mean Rewards", "Standard deviation", "Rewards(per episode)"])
        for ins in test_instances:
            results = results_all_instances[ins]
            mean_creward = results["creward_means"][0]
            std_creward = results["creward_stds"][0]
            eps_crewards = results["ep_crewards"][0]
            writer = csv.writer(file)
            writer.writerow([ins, mean_creward, std_creward, eps_crewards])
    print("Saved results on file " + csv_file)


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    helper.load_config(config_file)

    my_config.train_instance = ""
    if my_config.setting == "ippc":
        if my_config.domain == 'navigation':
            my_config.test_instance = ",".join([str(910+i) for i in range(40)])
        else:
            my_config.test_instance = ",".join([str(1110+i) for i in range(40)])
    elif my_config.setting == "lr":
        my_config.test_instance = ",".join([str(3200+i) for i in range(0, 200)])
        if my_config.domain == 'navigation':
            my_config.test_instance = ",".join([str(2200+i) for i in range(200)])
	
    model_dir, ckpt_dir, log_file = helper.get_model_dir(config_file)
    if my_config.restore_config:
        helper.restore_settings(model_dir)

    if my_config.exact_checkpoint is None:
        ckpts = helper.read_checkpoint_log(log_file)
        best_rew = -1000000
        for ckpt in ckpts:
            toks = ckpt.split(",")
            i = toks[0]
            toks = toks[1:]
            if my_config.setting == "lr":
                toks = toks[:100] # 100 val instances in lr
            elif my_config.setting == "ippc":
                toks = toks[:10] # 10 val instances in ippc
            rew = np.mean([float(x) for x in toks])
            if rew > best_rew:
                best_rew = rew
                my_config.exact_checkpoint = i

    print("Exact checkpoint:", my_config.exact_checkpoint)
    test(model_dir, ckpt_dir)
