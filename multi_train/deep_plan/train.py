#!/usr/bin/env python3
# =============================================================================

import sys
import os
import multiprocessing
import time
import pickle
import pdb
import random
import numpy as np
from datetime import datetime

import my_config

lock = multiprocessing.Lock()

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
network_path = os.path.abspath(os.path.join(curr_dir_path,"networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path
parent_dir_path = os.path.abspath(os.path.join(curr_dir_path,"..",".."))
if parent_dir_path not in sys.path:
    sys.path = [parent_dir_path] + sys.path

from policy_monitor import PolicyMonitor
import helper
from model_factory import ModelFactory

load_saved_dataset = False

# Performs one network update from the given batch (x, y). 
# Returns two values the policy loss and the aux loss (None if not enabled).
# @tf.function
def train_step(network, x, y, env_wrapper, loss_fn, optimizer, grad_clip_value, multiplier=0.0):
    import tensorflow as tf
    if my_config.add_aux_loss:
        with tf.GradientTape() as policynet_tape:
            policynet_pred, dist_attn_coef = network.policy_prediction(x, env_wrapper, return_attn_coef=True)
            policynet_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y, policynet_pred)
            if my_config.use_fluent_for_kl:
                random_node = tf.constant(env_wrapper.get_random_fluent_node(), dtype=tf.int32)
            else:
                random_node = tf.constant(env_wrapper.get_random_node(), dtype=tf.int32)
            
            random_heads = tf.constant(random.sample(list(np.arange(0, dist_attn_coef.shape[3])), k=2), dtype=tf.int32)
            attn_coef0 = dist_attn_coef[:,random_node,:,random_heads[0]] # shape-> BXN
            attn_coef1 = dist_attn_coef[:,random_node,:,random_heads[1]] # shape-> BXN

            aux_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(attn_coef0, attn_coef1)
            total_loss = policynet_loss - multiplier * aux_loss

        grads_of_policynet = policynet_tape.gradient(total_loss, network.trainable_variables)
        grads_of_policynet = tf.clip_by_global_norm(grads_of_policynet, grad_clip_value)

        optimizer.apply_gradients(zip(grads_of_policynet[0], network.trainable_variables))
        return policynet_loss, aux_loss

    else:
        with tf.GradientTape() as policynet_tape:
            policynet_pred = network.policy_prediction(x, env_wrapper)
            policynet_loss = loss_fn(y, policynet_pred)
        grads_of_policynet = policynet_tape.gradient(policynet_loss, network.trainable_variables)
        grads_of_policynet = tf.clip_by_global_norm(grads_of_policynet, grad_clip_value)

        optimizer.apply_gradients(zip(grads_of_policynet[0], network.trainable_variables))
        return policynet_loss, None

# Trains for the given number of epochs.
# Each epoch uses the entire dataset of each instance to perform updates.
def train(model_dir, ckpt_dir, log_file=None):
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    train_instances, val_instances = helper.get_instance_names()
    train_envs = helper.make_envs(train_instances)
    print("Envs created.")

    policynet_optim = tf.keras.optimizers.Adam(learning_rate=my_config.lr)
    model_factory = ModelFactory(train_envs[0], policynet_optim=policynet_optim)

    network = model_factory.create_network()
    network.init_network(train_envs[0])
    policy_monitor = PolicyMonitor(
        envs=helper.make_envs(val_instances),
        network=network,
        domain=my_config.domain,
        summary_writer=None,
        model_factory=model_factory)
    policy_monitor.network_copy.init_network(train_envs[0])
    print("Created policy monitor.")

    # Create CheckpointManager
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)

    # Load weights
    step = 0
    start_epoch = 0
    if my_config.use_pretrained:
        model_factory.load_ckpt(ckpt_num=None)
        print("Loaded model from checkpoint: " + str(model_factory.ckpt_manager.latest_checkpoint))
        step = network.trained_steps.numpy()
        start_epoch = network.trained_epochs.numpy()

    # SUPERVISED TRAINING STARTS
    # Training dataset
    from supervised_dataset import SupervisedDataset
    batch_size = my_config.batch_size # fixed at 32
    dataset_ob = SupervisedDataset(train_instances, train_envs, batch_size)
    print("Loading datasets.")

    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    grad_clip_value = model_factory.grad_clip_value
    best_val_reward = -float('inf')
    best_ckpt = ""
    ckpt_log = []
    for epoch in range(start_epoch, my_config.train_epochs):
        print("\n\n------Start of epoch %d" % (epoch,))
        for ins, env_index, states, actions in dataset_ob:
            # Get samples from the current instance's dataset
            total_size = len(states)
            cur_loc = 0

            # Uses all samples to train
            while cur_loc < total_size:
                # Take them in batches
                if cur_loc + batch_size < total_size:
                    x = states[cur_loc: cur_loc + batch_size]
                    y = actions[cur_loc: cur_loc + batch_size]
                    cur_loc += batch_size
                else:
                    x = states[cur_loc:]
                    y = actions[cur_loc:]
                    cur_loc = total_size+1

                kl_multiplier = 0
                if my_config.add_aux_loss:
                    if my_config.decay_aux_loss:
                        if step < 2000:
                            kl_multiplier = 0.1
                        elif 2000 <= step < 3000:
                            kl_multiplier = 0.1 * (3000 - step) / 1000
                    else:
                        kl_multiplier = 0.1

                start_time = time.time()
                # Each batch_size samples, do an update
                loss_value, aux_value = train_step(network, x, y, train_envs[env_index],
                    loss_fn=loss_fn,
                    optimizer=model_factory.policynet_optim,
                    grad_clip_value=grad_clip_value,
                    multiplier=kl_multiplier)
                ckpt_log.append((epoch, ins, time.time() - start_time, loss_value))
                step += 1
                network.trained_steps.assign(step)

            if my_config.add_aux_loss:
                print("Instance %d \t| Total steps: %d | Imitation loss: %.4f | KL loss: %.4f | KL multiplier: %.4f" % (ins, step-1, float(loss_value), float(aux_value), multiplier))
            else:
                print("Instance %d \t| Total steps: %d | Imitation loss: %.4f" % (ins, step-1, float(loss_value)))
        
        network.trained_epochs.assign(epoch)

        # Validation
        if (epoch % my_config.ckpt_freq) == my_config.ckpt_freq-1:
            policy_monitor.copy_params()
            save_path = model_factory.save_ckpt()
            time.sleep(5)
            results = policy_monitor.eval_policy(num_episodes=my_config.num_validation_episodes)
            if log_file is not None:
                rewards_str = ",".join([str(mr) for mr in results["creward_means"]])
                helper.write_content(log_file, f"{model_factory.get_ckpt_num()},{rewards_str}\n")
            # Log best and current mean total rewards.
            val_reward = np.mean(results["creward_means"])
            print(f"This checkpoint has a reward of {val_reward}, best is {best_val_reward}.")
            if not my_config.keep_ckpts:
                if val_reward <= best_val_reward:
                    os.remove(save_path + ".index")
                    os.remove(save_path + ".data-00000-of-00001")
                    print("Deleting this.")
                else:
                    if best_ckpt != "":
                        os.remove(best_ckpt + ".index")
                        os.remove(best_ckpt + ".data-00000-of-00001")
                    best_ckpt = save_path
                    print("Deleting previous.")
            best_val_reward = max(best_val_reward, val_reward)
            # Log loss and total rewards.
            with open(save_path + "_losses.csv", 'w') as f:
                f.write("epoch\tins\ttime\tloss\n")
                for (e, ins, t, loss) in ckpt_log:
                    f.write(f"{e}\t{ins}\t{t}\t{loss}\n")
            with open(save_path + "_rewards.csv", 'w') as f:
                f.write("ins\treward\tlength\ttime\n")
                for (env, crewards, lengths, times) in zip(policy_monitor.envs, results["ep_crewards"], results["ep_lengths"], results["ep_times"]):
                    for r, l, t in zip(crewards, lengths, times):
                        f.write(f"{env.get_instance_num()}\t{r}\t{l}\t{t}\n")
            ckpt_log = []

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    helper.load_config(config_file)
    
    #For each domain, we generate 1000 training, 100 validation, and 200 test instances with size increasing from train to val to test instances.
    if my_config.setting == "ippc":
        my_config.train_instance = ",".join(str(900+i) for i in range(200))   
        my_config.test_instance = ",".join(str(1100+i) for i in range(10))

        # Some training files weren't created properly which is why they've been excluded from training
        if my_config.domain == 'navigation':
            my_config.train_instance = ",".join([str(700+i) for i in range(200) if 700+i not in [705,798]])
            my_config.test_instance = ",".join(str(900+i) for i in range(10))
        
        if my_config.domain == 'triangle_tireworld':
            my_config.train_instance = ",".join(str(1000+i) for i in range(100)) # Generator doesn't have enough diversity in parameters, only need a 100 instances
            my_config.test_instance = ",".join(str(1100+i) for i in range(10))
    
    elif my_config.setting == "lr":
        if my_config.domain == 'recon': # SRecon
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2177])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'academic_advising_prob': # EAcad
            # These instances were rejected because their score is worse than a no-op policy
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2100,2119,2126,2130,2143,2150,2183,2185,2189,2194,2203,2205,2242,2258,2283,2290,2296,2329,2336,2344,2351,2417,2432,2472,2482,2488,2505,2508,2523,2530,2540,2569,2586,2589,2597,2605,2613,2623,2640,2686,2730,2747,2778,2780,2869,2889,2961,2994,2998,3023,3042,3081,3095])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'pizza_delivery_windy': # Pizza
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2692])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))        
        if my_config.domain == 'navigation': # DNav
            my_config.train_instance = ",".join([str(1100+i) for i in range(1000) if 1100+i not in [1333,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979]])
            my_config.test_instance = ",".join([str(2100+i) for i in range(100)])
        if my_config.domain == 'stochastic_wall': #StWall
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2806,2835,2837,3069,3080])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'corridor': # StNav
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2211])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))

    model_dir, ckpt_dir, log_file = helper.get_model_dir(config_file, create=True)
    if my_config.restore_config:
        helper.restore_settings(model_dir)

    print("Domain: ", my_config.domain)
    print("Model dir: ", my_config.model_dir)
    print(my_config.train_instance, my_config.test_instance)
    train(model_dir, ckpt_dir, log_file)
