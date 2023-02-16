import argparse
import importlib
import shutil
import datetime
import numpy as np
import itertools
import torch
import time
import csv
import os
from uav_sac.verify import verify_models, generate_agent_simulator
from uav_sac.sac import SAC
from uav_sac.training_config import training_config_from_json
from uav_sac.replay_memory import ReplayMemory
from uav_sac.utils import load_random_animation

parser = argparse.ArgumentParser(description="Pytorch SAC Training Session")
parser.add_argument('cfg_file', type=str, default='./training_cfg.json', help="training config file")
args = parser.parse_args()

cfg = training_config_from_json(args.cfg_file)


torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# Environment
env_cls = getattr(importlib.import_module("uav_sac.environments." + ".".join(cfg.env_name.split(".")[:-1])), cfg.env_name.split(".")[-1])
env = env_cls(load_random_animation(), cfg)

# Agent
# expert_agent = SAC(env.obs_state_len, env.action_space, args)
# expert_agent.load_checkpoint('winning_config_c3/c3_model')

agent = SAC(env.obs_state_len, env.action_space, cfg.hyperparams, map_input=(env.observation_space.shape[2],
                                                                  env.observation_space.shape[0]-1,
                                                                  env.observation_space.shape[1]))
run_dir = os.path.join(cfg.repo_path, "runs", 
                       "{}_SAC_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                 cfg.env_name.split('.')[-1], cfg.hyperparams.policy))
os.mkdir(run_dir)
shutil.copyfile(args.cfg_file, f'{run_dir}/{os.path.basename(args.cfg_file)}')

with open(f"{run_dir}/training_results.csv", 'w') as results:
    results_csv = csv.writer(results, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL,
                             quotechar="|")
    results_csv.writerow(['avg_reward', 'crash_rate (%)', 'delta error', 'time_elapsed (s)'])


def save_results(avg_reward, crash_rate, delta_error, time_elapsed):
    with open(f"{run_dir}/training_results.csv", 'a') as results:
        results_csv = csv.writer(results, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL,
                                 quotechar="|")
        results_csv.writerow([avg_reward, crash_rate, delta_error, time_elapsed])


with open(f"{run_dir}/training_loss.csv", 'w') as losses:
    loss_csv = csv.writer(losses, delimiter=',',
                          quoting=csv.QUOTE_MINIMAL,
                          quotechar="|")
    loss_csv.writerow(['critic1_loss', 'critic2_loss',
                       'policy_loss', 'ent_loss', 'alpha'])


def save_losses(critic1_loss, critic2_loss, policy_loss, ent_loss, alpha):
    with open(f"{run_dir}/training_loss.csv", 'a') as losses:
        loss_csv = csv.writer(losses, delimiter=',',
                              quoting=csv.QUOTE_MINIMAL,
                              quotechar="|")
        loss_csv.writerow([critic1_loss, critic2_loss,
                           policy_loss, ent_loss, alpha])

start_time = time.time()

# Memory
memory = ReplayMemory(cfg.episode.replay_size, cfg.seed)

# Training Loop
total_numsteps = 0
updates = 0
if cfg.episode.updates_per_step < 1:
    steps_per_update = int(1/cfg.episode.updates_per_step)
else:
    steps_per_update = None

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    while not done:
        if cfg.episode.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > cfg.hyperparams.batch_size:
            # Number of updates per step in environment
            if steps_per_update:
                if episode_steps % steps_per_update == 0:
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, cfg.hyperparams.batch_size, updates)
                    save_losses(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha)
                    updates += 1
                else:
                    for i in range(int(cfg.episode.updates_per_step)):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, cfg.hyperparams.batch_size, updates)
                        save_losses(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha)
                        updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

    if total_numsteps > cfg.episode.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % cfg.episode.eval == 0 and cfg.episode.eval != 0:
        simulator = generate_agent_simulator(agent, cfg.horizon)
        report = verify_models(cfg, simulator, save_path=f"{run_dir}/{i_episode}_", display=False)
        save_results(report.average_reward, report.crash_rate, report.average_delta_error, time.time() - start_time)

        print("----------------------------------------")
        print("Test Episodes: {}, Total updates {}, Avg. Reward: {}, Crash Rate: {}".format(cfg.episode.verification_episodes, updates, round(report.average_reward, 5), report.crash_rate))
        print("----------------------------------------")
        agent.save_checkpoint(cfg.env_name, ckpt_path=f"{run_dir}/{i_episode}_model")

env.close()
