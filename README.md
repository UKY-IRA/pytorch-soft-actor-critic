### Description
------------
Forked from [Pytorch SoftActor Critic Gym Example](https://github.com/pranz24/pytorch-soft-actor-critic).

Reimplementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) and a deterministic variant of SAC from [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

Specialized environment made to support UAV autonomus trajectory planning for aerial gas leak mapping. Uses an adaptation of (pompy)[https://github.com/InsectRobotics/pompy] data recordings for generated gas maps.

### Requirements
------------
The repo is setup to work well with conda environments. See [this docsite](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) for how to install conda.

Once conda is installed you need to install the required packages by pointing conda to torch_env.yml.

```sh
# all script snippets start at the top of the repo
cd uav_sac
conda --prefix ../env -f torch_env.yml
```

Note: sometimes the version solves unexpectedly fail. You can remove the strict versioning from each package in the yaml file and retry.

### Default Configuration and Usage
------------
### Usage

## Generating training datasets
We need to generate a LOT of example datasets of gas plume dispersion to train on. Realistically you should generate as much as your storage allows.

```sh
mkdir animations
./uav_sac/scripts/generate_animations 1000 # or however many you can
```


## Training
Run Locally:
```sh
conda activate
cd uav_sac
python3 main.py training_config.json
```

Run through the singularity job dispatcher:
```sh 
conda activate
pip3 install .  # need to install our changes into the environment
./uav_sac/scripts/submit.sh
```

## Verification


### Arguments
------------
```
PyTorch Soft Actor-Critic Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: HalfCheetah-v2)
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --eval EVAL           Evaluates a policy a policy every 10 episode (default:
                        True)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 5e-3)
  --lr G                learning rate (default: 3e-4)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  --seed N              random seed (default: 123456)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1e6)
  --cuda                run on CUDA (default: False)
```
