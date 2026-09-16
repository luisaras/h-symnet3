import os, sys, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curr_dir_path,"..",".."))
if parent_dir_path not in sys.path:
    sys.path = [parent_dir_path] + sys.path

from .policy_monitor import PolicyMonitor
from .model_factory import ModelFactory
from . import helper
from . import my_config

def create_model_factory(ckpt_dir, env):
    import tensorflow as tf
    policynet_optim = tf.keras.optimizers.Adam(learning_rate=my_config.lr)
    #policynet_optim = tf.keras.optimizers.RMSprop(learning_rate=my_config.lr, rho=0.99, momentum=0.0, epsilon=1e-6)
    model_factory = ModelFactory(env, policynet_optim=policynet_optim)
    # Init network
    network = model_factory.create_network()
    network.init_network(env)
    model_factory.set_ckpt_network(ckpt_dir, network)
    return network, model_factory

def create_policy_monitor(ckpt_dir, test_envs):
    network, model_factory = create_model_factory(ckpt_dir, test_envs[0])
    # Load weights
    model_factory.load_ckpt(my_config.exact_checkpoint)
    policy_monitor = PolicyMonitor(
        envs=test_envs,
        network=network,
        domain=my_config.domain)
    print("Created policy monitor.")
    return policy_monitor

def evaluate_ckpt(ckpt_dir, test_instance, num_episodes, process_index):
    test_envs = helper.make_envs([test_instance])
    print("Envs created.")
    policy_monitor = create_policy_monitor(ckpt_dir, test_envs)
    print("In process:", process_index)
    results = policy_monitor.eval_policy(num_episodes=num_episodes)
    test_envs[0].close()
    sys.stdout = sys.__stdout__
    print("Rewards:", results["ep_crewards"])
    return results

def evaluate_all(all_args, eval_func=None):
    if eval_func is None:
        eval_func = evaluate_ckpt
    results_all_instances = {}
    with ThreadPoolExecutor(max_workers=my_config.num_threads) as executor:
        futures = {executor.submit(eval_func, *arg): i for i, arg in all_args.items()}
        for future in as_completed(futures.keys()):
            instance = futures[future]
            try:
                results_all_instances[instance] = future.result()
            except Exception as exc:
                print(f"Task {instance} generated an exception: {exc}", file=sys.stderr)
                traceback.print_exc()
                sys.exit(1)
    return results_all_instances
