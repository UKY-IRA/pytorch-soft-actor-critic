import numpy as np
from model_plane import GaussianPolicy
from plane_env import Plane
from sac import SAC
import torch
import argparse
import matplotlib.pyplot as plt
import json

def verify_models(args, agent, verification_eps, save_path=False, display=False):
    envs = []
    for e in range(args.num_planes):
        env = Plane()
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        envs.append(env)

    rewards = []
    crashed = 0
    avg_reward = 0
    for n in range(verification_eps):
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
            if display:
                plt.show()
            elif save_path:
                plt.savefig(save_path.format(n))
            plt.clf()
    avg_reward /= verification_eps
    return avg_reward, crashed/(verification_eps*3)

def main():
    parser = argparse.ArgumentParser(description='Pytorch Plane Model Verification Args')
    parser.add_argument('--conf', required=True, nargs='+',
                        help='config file for the training run, required for model creation')
    parser.add_argument('--model_name', required=True,
                        help='model to run verification on')
    args = parser.parse_args()
    for conf_fname in args.conf:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args()

    agent = SAC(Plane.obs_state_len, Plane.action_space, args)
    agent.load_checkpoint(args.model_name)
    episodes = 11
    avg_reward, crashed = verify_models(args, agent, episodes, save_path="./verification_{}.png", display=False)
    print(f"Average Reward over {episodes} runs {avg_reward}, crash rate {crashed}")

if __name__ == '__main__':
    main()
