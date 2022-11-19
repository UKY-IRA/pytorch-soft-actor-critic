import numpy as np
import math
from networks.conv2d_model import GaussianPolicy, QNetwork
from environments.simple2duav import Simple2DUAV
from environments.belief2d import Belief2D
from sac import SAC
import torch
import copy
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json

font_dirs = ['/home/jaas224/fonts_for_figures']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 22}

plt.rc('font', **font)


def display_results(result_dict, n, title="Simple2DUAV Trajectory", display=False, save_path=None):
    start_image = result_dict['start_image']
    final_image = result_dict['final_image']
    trajectories = result_dict['trajectory']

    # plot the initial map
    mesh = plt.pcolormesh(np.array(start_image).T, cmap="RdYlGn", alpha=0.2)
    # cbar = plt.colorbar(mesh)
    # cbar.set_label("Value (Iz(s))")
    plt.title("Starting Value Map")
    plt.xlabel("Position (X)")
    plt.ylabel("Position (Y)")
    plt.xlim([0, Simple2DUAV.ydim])
    plt.ylim([0, Simple2DUAV.xdim])
    plt.clim(0, 1)
    if display:
        plt.show()
    elif save_path:
        plt.savefig(f"{save_path}startmap_{n}.png",
                    dpi=300,
                    bbox_inches="tight")
    plt.clf()

    # plot the trajectory
    mesh = plt.pcolormesh(np.array(final_image).T, cmap="RdYlGn", alpha=0.2)
    cbar = plt.colorbar(mesh)
    cbar.set_label("Confidence P(C(x,y))")
    plt.title(title)
    plt.xlabel("Position (X)")
    plt.ylabel("Position (Y)")
    plt.xlim([0, Simple2DUAV.ydim])
    plt.ylim([0, Simple2DUAV.xdim])
    plt.clim(0, 1)
    for i, p in enumerate(trajectories):
        plt.scatter(p[0][1], p[0][0], marker='o')
        plt.plot(np.array(p).T[1], np.array(p).T[0], label="{i}")
    if display:
        plt.show()
    elif save_path:
        plt.savefig(f"{save_path}{n}.png", dpi=300, bbox_inches="tight")
    plt.clf()


def generate_agent_simulator(agent, horizon):
    def _run(planes):
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = [[] for _ in range(len(planes))]
        while not done:
            # reset turns
            planes[list(planes.keys())[0]].bspace.step(Simple2DUAV.dt*horizon) # step the map at the start of each turn
            [p._set_state_vector() for p in planes.values()]
            turns = [(i,p) for i, p in planes.items()] # [(plane_index, plane_env)]
            while len(turns) > 0:
                # pick a winning plane to take its turn first
                qs = agent.get_vs([p.normed_state() for _, p in turns]) # pass all env states as batch
                winner = np.argmax(qs)
                select_action = lambda: agent.select_action(turns[winner][1].normed_state(), evaluate=True)  # Sample action from policy

                # simulate the plane forward {horizon} steps
                for _ in range(horizon):
                    action = select_action()
                    next_state, reward, plane_done, _ = turns[winner][1].step(action) # Step
                    episode_reward += reward
                    if plane_done:
                        planes.pop(turns[winner][0])
                        if not math.isclose(turns[winner][1].t, Simple2DUAV.maxtime, rel_tol=2*Simple2DUAV.dt):
                            crashed += 1
                        done = len(planes) == 0
                        break
                    plane_trajs[turns[winner][0]].append([turns[winner][1].x, turns[winner][1].y])
                # remove plane when turn is completed
                turns.pop(winner)
        return plane_trajs, episode_reward, crashed
    return _run


def generate_greedy_simulator():
    def _run(planes):
        Simple2DUAV.dt = 0.4
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = [[] for _ in range(len(planes))]
        action_set = np.arange(-Simple2DUAV.action_max+0.0000001, Simple2DUAV.action_max-0.0000001, math.pi/36)
        while not done:
            turns = [(i,p) for i,p in planes.items()]
            planes[list(planes.keys())[0]].bspace.step(Simple2DUAV.dt) # step the map at the start of each turn
            [p._set_state_vector() for p in planes.values()]
            for i, plane in turns:
                plane._set_state_vector()
                rs = []
                for a in action_set:
                    current_state = np.copy(plane.state)
                    _, r, _, _ = plane.step(a)
                    rs.append(r)
                    plane.reset_state_from(current_state)
                action = action_set[np.argmax(np.array(rs))]
                next_state, reward, plane_done, _ = plane.step(action)
                episode_reward += reward
                if plane_done:
                    planes.pop(i)
                    if not math.isclose(plane.t, plane.maxtime, rel_tol=2*Simple2DUAV.dt):
                        crashed += 1
                plane_trajs[i].append([plane.x, plane.y])
            done = len(planes) == 0
        Simple2DUAV.dt = 0.2
        return plane_trajs, episode_reward, crashed
    return _run


def generate_fixed_simulator():
    def _run(planes):
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = [[] for _ in range(len(planes))]

        ydir = {}
        xpad = 4
        xmin = xpad
        xmax = Simple2DUAV.xdim - xpad
        for i, plane in planes.items():
            if plane.y > plane.ydim/2:
                ydir[i] = 'down'
            else:
                ydir[i] = 'up'

        right = -2*math.pi/18
        soft_right = -1*math.pi/36
        left = 2*math.pi/18
        soft_left = 1*math.pi/36

        while not done:
            turns = [(i, p) for i, p in planes.items()]
            planes[list(planes.keys())[0]].bspace.step(Simple2DUAV.dt)  # step the map at the start of each turn
            [p._set_state_vector() for p in planes.values()]
            for i, plane in turns:
                if ydir[i] == 'up' and plane.x > xmax and not math.isclose(plane.yaw, math.pi, rel_tol=0.05): # turn right
                    action = right
                elif ydir[i] == 'up' and plane.x < xmin and not math.isclose(plane.yaw, 0, rel_tol=0.05): # turn left
                    action = left
                elif ydir[i] == 'down' and plane.x > xmax and not math.isclose(plane.yaw, math.pi, rel_tol=0.05): # turn right
                    action = right
                elif ydir[i] == 'down' and plane.x < xmin and not math.isclose(plane.yaw, 0, rel_tol=0.05): # turn left
                    action = left
                else:
                    action = 0
                    # keep near pi
                    if plane.yaw > math.pi/2 and plane.yaw < math.pi: # left
                        action = soft_left
                    elif plane.yaw < 3*math.pi/2 and plane.yaw > math.pi: # right
                        action = soft_right
                    # keep near 0
                    elif plane.yaw < math.pi/2 and plane.yaw > 0: # right
                        action = soft_right
                    elif plane.yaw > 3*math.pi/2 and plane.yaw < math.pi*2: # left
                        action = soft_left
                next_state, reward, plane_done, _ = plane.step(action)
                episode_reward += reward
                if plane_done:
                    planes.pop(i)
                    if not math.isclose(plane.t, plane.maxtime, rel_tol=2*Simple2DUAV.dt):
                        crashed += 1
                plane_trajs[i].append([plane.x, plane.y])
            done = len(planes) == 0
        return plane_trajs, episode_reward, crashed
    return _run

def verify_models(gamma, num_planes, verification_eps, simulator, save_path=False, display=False):
    envs = []
    for e in range(num_planes):
        env = Simple2DUAV(gamma)
        envs.append(env)

    rewards = []
    results = []
    crashed = 0
    avg_reward = 0
    for n in range(verification_eps):
        # reset the envs
        result = {}
        belief_space = Belief2D(Simple2DUAV.xdim, Simple2DUAV.ydim)
        for e in envs:
            e.reset(belief_space=belief_space)
            e._set_state_vector()

        if n % 10 == 0:
            belief = np.take(belief_space.img, 2, axis=2)
            result['start_image'] = copy.copy(belief.T).tolist()

        # total_value = np.sum(belief_space.img)
        planes = {j: env for j, env in enumerate(envs)}  # preserves index even after deletion
        plane_trajs, episode_reward, crash = simulator(planes)
        crashed += crash
        rewards.append(episode_reward)  #/total_value) # norm result

        if n % 10 == 0:
            belief = np.take(belief_space.img, 2, axis=2)
            result['final_image'] = copy.copy(belief.T).tolist()
            result['score'] = episode_reward
            result['trajectory'] = copy.copy(plane_trajs)
            results.append(result)
            display_results(result, n, display=display, save_path=save_path)
    with open(f"{save_path}results.json", 'w') as outfile:
        json.dump(results, outfile)
    avg_reward = sum(rewards)/verification_eps
    stdev_reward = (sum([((x - avg_reward) ** 2) for x in rewards]) / len(rewards)) ** 0.5
    return avg_reward, stdev_reward, crashed/(verification_eps*num_planes)

def compare_simulators(gamma, simulators, save_path=False, display=False):
    env = Simple2DUAV(gamma)

    results = []
    for n in range(5):
        # reset the envs
        result = {}
        belief_space = Belief2D(Simple2DUAV.xdim, Simple2DUAV.ydim)
        env.reset(belief_space)

        belief = np.take(belief_space.img, 2, axis=2)
        result['start_image'] = copy.copy(belief.T).tolist()
        start_state = copy.copy(env.state).tolist()

        for sim_name, simulator in simulators.items():
            env.reset_state_from(np.array(start_state))
            plane = {0: env}
            plane_trajs, episode_reward, _ = simulator(plane)

            belief = np.take(belief_space.img, 2, axis=2)
            result['final_image'] = copy.copy(belief.T).tolist()
            result['score'] = episode_reward
            result['trajectory'] = copy.copy(plane_trajs)
            results.append(result)
            display_results(result, n, display=display, save_path=f"{save_path}{sim_name}_", title=f"{sim_name} Trajectory")

def main():
    parser = argparse.ArgumentParser(description='Pytorch Simple2DUAV Model Verification Args')
    parser.add_argument('-c', '--config', required=True, nargs='+',
                        help='config file for the training run, required for model creation')
    parser.add_argument('-m', '--model_path', required=True,
                        help='model to run verification on')
    args = parser.parse_args()
    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args()

    env = Simple2DUAV(args.gamma)
    agent = SAC(env.obs_state_len, env.action_space, args,
                map_input=(env.observation_space.shape[2],
                           env.observation_space.shape[0]-1,
                           env.observation_space.shape[1]))
    agent.load_checkpoint(args.model_path)

    simulators = {
        "SAC": generate_agent_simulator(agent, args.horizon),
        "SAC (horizon=1)": generate_agent_simulator(agent, 1),
        "Greedy": generate_greedy_simulator(),
        "Fixed": generate_fixed_simulator()
    }
    compare_simulators(args.gamma, simulators, save_path="current_verification/", display=False)
    '''
    episodes = 101
    for num_planes in range(1,4):
        print(f"Number of planes: {num_planes}")
        simulator = generate_agent_simulator(agent, 10)
        avg_reward, stdev, crashed = verify_models(args.gamma, num_planes, episodes, simulator, save_path="current_verification/", display=False)
        print(f"Agent average reward over {episodes} runs {avg_reward}/{stdev}, crash rate {crashed}")

        simulator = generate_agent_simulator(agent, 1)
        avg_reward, stdev, crashed = verify_models(args.gamma, num_planes, episodes, simulator, save_path="current_verification/", display=False)
        print(f"Short horizon agent average reward over {episodes} runs {avg_reward}/{stdev}, crash rate {crashed}")

        simulator = generate_greedy_simulator()
        avg_reward, stdev, crashed = verify_models(args.gamma, num_planes, episodes, simulator, save_path="current_verification/", display=False)
        print(f"Greedy average reward over {episodes} runs {avg_reward}/{stdev}, crash rate {crashed}")

        simulator = generate_fixed_simulator()
        avg_reward, stdev, crashed = verify_models(args.gamma, num_planes, episodes, simulator, save_path="current_verification/", display=False)
        print(f"Fixed average reward over {episodes} runs {avg_reward}/{stdev}, crash rate {crashed}")

    '''
if __name__ == '__main__':
    main()
