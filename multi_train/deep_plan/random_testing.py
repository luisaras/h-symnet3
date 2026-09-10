import helper
import my_config
import numpy as np

test_instances = [str(i) for i in range(910, 950)]
def test(domain='navigation', instances=test_instances, num_episodes=30):
    my_config.domain = domain
    envs = helper.make_envs(instances)
    writer = open(f'result_scripts/random_results/{domain}_random.csv', 'a')
    writer.write("Instance,Mean Reward\n")
    writer.close()

    for i, inst in enumerate(instances):
        rewards_i = []
        envs[i].seed(0)
        for j in range(num_episodes):
            initial_state, done = envs[i].reset()
            state = initial_state
            episode_length = 0
            episode_reward = 0
            while not done:
                action = envs[i].get_random_action()
                previous_action = action
                next_state, reward, done, _ = envs[i].step(action)
                print(('state: {}  action: {}  reward: {} next: {}'.format(state, action, reward, next_state)))

                episode_reward += reward
                state = next_state

            rewards_i.append(episode_reward)
            print(('Episode Reward: {}'.format(episode_reward)))
            print()

        mean_total_reward = np.mean(rewards_i)
        print(inst, mean_total_reward)
        with open(f'result_scripts/random_results/{domain}_random.csv', 'a') as f:
            f.write(f"{inst},{mean_total_reward}\n")
        envs[i].close()

if __name__ == '__main__':
    test(sys.argv[1], [sys.argv[2]])