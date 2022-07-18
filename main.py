import argparse
import copy
import datetime
import gym
import numpy as np
import itertools
import torch
import csv
import os
import json
from plane_env import Plane
from sac import SAC
from verify import verify_models, generate_agent_simulator
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=int, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=float, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--num_planes', type=int, default=1, metavar='N',
                    help='number of planes to use in verification (default: 1)')
parser.add_argument('--horizon', type=int, default=10, metavar='N',
                    help='number of actions to plan ahead before moving on to the next plane')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
env = Plane(args.gamma)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
# expert_agent = SAC(env.obs_state_len, env.action_space, args)
# expert_agent.load_checkpoint('winning_config_c3/c3_model')

agent = SAC(env.obs_state_len, env.action_space, args, map_input=(env.observation_space.shape[2],
                                                                  env.observation_space.shape[0]-1,
                                                                  env.observation_space.shape[1]))
run_dir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                        args.policy, "autotune" if args.automatic_entropy_tuning else "")
os.mkdir(run_dir)

reward_file = csv.writer(open(f"{run_dir}/rewards.csv", 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL, quotechar="|")
reward_file.writerow(['avg_reward', 'crash_rate'])
loss_file = csv.writer(open(f"{run_dir}/training_loss.csv", 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL, quotechar="|")
loss_file.writerow(['critic1_loss', 'critic2_loss', 'policy_loss', 'ent_loss', 'alpha'])
with open(f'{run_dir}/run_args.cfg', 'w') as conf:
    conf.write(json.dumps(vars(args),  indent=4, sort_keys=True))
# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
if args.updates_per_step < 1:
    steps_per_update = int(1/args.updates_per_step)
else: steps_per_update = None

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            if steps_per_update:
                if episode_steps % steps_per_update == 0:
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    loss_file.writerow([critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha])
                    updates += 1
                else:
                    for i in range(int(args.updates_per_step)):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                        loss_file.writerow([critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha])
                        updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % args.eval == 0 and args.eval != 0:
        episodes = 21
        simulator = generate_agent_simulator(agent, args.horizon)
        avg_reward, _, crashed = verify_models(args.gamma, args.num_planes, episodes, simulator, save_path=f"{run_dir}/{i_episode}_", display=False)
        reward_file.writerow([avg_reward, crashed])

        print("----------------------------------------")
        print("Test Episodes: {}, Total updates {}, Avg. Reward: {}, Crash Rate: {}".format(episodes, updates, round(avg_reward, 5), crashed))
        print("----------------------------------------")
        agent.save_checkpoint(args.env_name, ckpt_path=f"{run_dir}/{i_episode}_model")

env.close()

