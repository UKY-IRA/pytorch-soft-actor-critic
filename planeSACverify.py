import numpy as np
from model_plane import GaussianPolicy
from plane_env import Plane
from sac import SAC
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--model_name', default="winning_config_c3/c3_model",
                    help='model file')
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
parser.add_argument('--num_planes', type=int, default=3, metavar='N',
                    help='number of planes to simulate')
parser.add_argument('--horizon', type=int, default=10, metavar='N',
                    help='number of actions to plan ahead before moving on to the next plane')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()
envs = []
for e in range(args.num_planes):
    env = Plane()
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    envs.append(env)
agent = SAC(env.obs_state_len, env.action_space, args)
agent.load_checkpoint('winning_config_c3/c3_model')



rewards = []
crashed = 0
avg_reward = 0
episodes = 50
for n in range(episodes):
    # reset the envs
    envs[0].reset()
    global_map = envs[0].image
    for e in envs:
        e.reset()
        e.image = global_map
        e._set_state_vector()
    planes = {j:env for j, env in enumerate(envs)} # preserves index even after deletion
    episode_reward = 0
    done = False

    plane_trajs = [[] for _ in range(len(planes))]
    while not done:
        # reset turns
        turns = [(i,p) for i, p in planes.items()] # [(plane_index, plane_env)]
        while len(turns) > 0:
            # pick a winning plane to take its turn first
            qs = agent.get_vs([p.normed_state() for _, p in turns]) # pass all env states as batch
            winner = np.argmax(qs)
            select_action = lambda: agent.select_action(turns[winner][1].normed_state(), evaluate=True)  # Sample action from policy

            # simulate the plane forward {horizon} steps
            for _ in range(args.horizon):
                action = select_action()
                next_state, reward, plane_done, _ = turns[winner][1].step(action) # Step
                episode_reward += reward
                if plane_done:
                    done_plane = turns[winner][0]
                    planes.pop(turns[winner][0])
                    if turns[winner][1].t < turns[winner][1].maxtime:
                        crashed += 1
                    done = len(planes) == 0
                    break
                if n % 10 == 0:
                    plane_trajs[turns[winner][0]].append(np.array([next_state[0][0]*Plane.xdim, next_state[0][1]*Plane.ydim]))
            # synchronize images
            global_map = turns[winner][1].image
            for plane in planes.values():
                plane.image = global_map
                plane._set_state_vector()
            # remove plane when turn is completed
            turns.pop(winner)
    avg_reward += episode_reward
    if n % 10 == 0:
        plt.pcolormesh(global_map.T, cmap="RdYlGn", alpha=0.2)
        for i,p in enumerate(plane_trajs):
            plt.scatter(p[0][0], p[0][1], marker='o')
            plt.title(f"Total Reward: {episode_reward}")
            plt.plot(np.array(p).T[0], np.array(p).T[1], label="{i}")
        plt.show()
avg_reward /= episodes
print(f"Average Reward over {episodes} runs {avg_reward}, crash rate {crashed/(episodes*3)}")
