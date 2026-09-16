#!/usr/bin/env python3
# =============================================================================
import os, sys, csv

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curr_dir_path,"..",".."))
if parent_dir_path not in sys.path:
    sys.path = [parent_dir_path] + sys.path

from multi_train.supervised import *


def test(model_dir, ckpt_dir):
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use GPU for inference

    _, test_instances = helper.get_instance_names()
    num_episodes = my_config.num_testing_episodes

    all_args = {instance: (ckpt_dir, instance, num_episodes, i) for i, instance in enumerate(test_instances)}
    results_all_instances = evaluate_all(all_args)

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
	
    model_dir, ckpt_dir, log_file = helper.get_model_dir(config_file)
    if my_config.restore_config:
        helper.restore_settings(model_dir)

    if my_config.exact_checkpoint is None:
        my_config.exact_checkpoint = helper.find_best_checkpoint(log_file)

    print("Exact checkpoint:", my_config.exact_checkpoint)
    test(model_dir, ckpt_dir)
