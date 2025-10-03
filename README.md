
![vid](https://github.com/user-attachments/assets/16bf3558-5c87-4749-ba4b-c96c6ab5a955)

# JSAC
JSAC is a Soft Actor-Critic (SAC) based Reinforcement Learning (RL) system designed for high-performance and stable learning from images. JSAC is well-suited for real-world, vision-based robotic tasks and learning from scratch. JSAC supports fully onboard vision-based RL training in edge devices such as Jetson Orin Nano. 

### JSAC Features
- High-performance SAC-based learning framework.  
- Improved stability for learning from images and learning from scratch.  
- Significant learning speed enabled by the JAX/Flax based implementation.  
- Asynchronous environment interaction, learning updates, and replay buffer sampling processes enable efficient resource usage for real-world tasks.  
- Supports full checkpointing, including replay buffer saving. Ideal for continuing learning on battery-powered real-world robots.
- The Codebase is designed for simplicity and improved readability.

To understand how JSAC works, please read the technical documentation: **[JSAC TECHNICAL DOCUMENTATION](https://github.com/fahimfss/JSAC/blob/main/extra/JSAC_Technical_Writeup.pdf)**

## Simulated Experiments

### Non-visual Tasks: Mujoco Environments

The following plot compares the performance of JSAC with [Stable-Baselines3 - SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) (sb3) and RedQ ([Randomized Ensembled Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/abs/2101.05982), [code implementation](https://github.com/ikostrikov/jaxrl)). The lines show the average return over 15 seeds, and the shaded regions show the confidence interval.

JSAC with synchronous learning updates (one update per environment step) is used in these tasks. The full hyperparameter settings used for JSAC can be viewed [here](https://github.com/fahimfss/JSAC/blob/main/hyperparameters.md#non-visual-tasks-mujoco-environments).
<p align="center" width="100%">
  <a href="url"><img src="https://github.com/user-attachments/assets/cbabc8c8-cd3b-477f-a487-b345d00c764e" height="90%" width="90%" ></a>
</p>
<!-- ![combined](https://github.com/user-attachments/assets/cbabc8c8-cd3b-477f-a487-b345d00c764e) -->

### Image-only Tasks: DeepMind Control (DMC) Environments
The following plot compares JSAC with [Dreamer-v3](https://github.com/danijar/dreamerv3), and [TDMPC-2](https://github.com/nicklashansen/tdmpc2). The Dreamer-v3 and TDMPC-2 results are collected from [here](https://github.com/nicklashansen/tdmpc2/tree/main/results). JSAC results are averaged over 15 seeds, while the Dreamer-v3 and TDMPC-2 results are averaged over 3 seeds where available from the source data.

For these tasks, JSAC was configured to use synchronous learning updates. The full hyperparameter settings used for these tasks can be viewed [here](https://github.com/fahimfss/JSAC/blob/main/hyperparameters.md#image-only-tasks-deepmind-control-dmc-environments).
<p align="center" width="100%">
  <a href="url"><img src="https://github.com/user-attachments/assets/e696e3c8-a906-4b1b-a043-9cdb996e8c20" height="90%" width="90%" ></a>
</p>
<!-- ![combined](https://github.com/user-attachments/assets/e696e3c8-a906-4b1b-a043-9cdb996e8c20) -->

<br>
Overall, JSAC mostly performs similarly or better compared to the state-of-the-art RL systems in simulated tasks.

## Real-World Experiment
### Create2-Orin Reacher Task
**Create2-Orin Setup:** For the Create2-Orin task, **learning from scratch was performed completely onboard a Jetson Orin Nano 8GB device, using only a portable battery pack for power**. A camera attached to the Create2 robot provided visual information to the JSAC agent. The task is to reach one of the two stickers (pink or green) attached to the Create2 arena walls (Check the video below!). The parameters are listed [here](https://github.com/fahimfss/JSAC/blob/main/hyperparameters.md#create2-orin-reacher-task), and the task file for training is [here](https://github.com/fahimfss/JSAC/blob/main/tasks/robotics/task_create2_orin_reacher_multi.py).

<a href="url"><img src="https://github.com/user-attachments/assets/2949f7cf-87ef-4be2-8898-005fa1538122" height="280" width="240" ></a>

The following plot shows the learning performance (single run on real hardware):
<a href="url"><img src="https://github.com/user-attachments/assets/88d77e48-c727-414b-8c8c-eef04baef900" height="50%" width="50%" ></a>

Here's a video of the trained agent after 75K steps of learning: 
  
https://github.com/user-attachments/assets/04e36b87-990e-452a-8497-185bcaae5954

<details>
<summary><h4>Continuing Real-World Training [+]</h4></summary>
  
In this Create2-Orin run, the training was stopped at 35K steps, and a checkpoint of the full training (network parameters and replay buffer) was created. The Create2 was shut down and disconnected from the battery pack for a minute. After that, training was resumed from the saved checkpoints, and the agent was trained to 75K steps. 

To resume training a run, first use the following hyperparameters to checkpoint:
- Set `replay_buffer_capacity` to the full size of the replay buffer (you should not change this value afterwards).
- Set `env_steps` to the number of steps you want the run to pause after.
- Set `save_model` to `True` and `save_model_freq` to a divisor of `env_steps`.
- Set `buffer_save_path` to a path.

For the Create2 Orin, these hyperparameters were set to the following values:
- `replay_buffer_capacity = 75,000`
- `env_steps = 35,000`
- `save_model = True`
- `save_model_freq = 5,000`
- `buffer_save_path = ./buffers/`

For resuming, set the following hyperparameter values (the other hyperparameter values should remain unchanged): 
- `env_steps = 75,000`
- `load_model = 35,000`
- `buffer_load_path = ./buffers/`

Also, most importantly, for the prompt `The work directory already exists...`, press **'Enter'** to resume the run. 
</details>

#### Real-World Experiments With Franka Panda Arm and UR5 Robot: Coming Soon!

## Installation
<details>
<summary><h4>Orin (from scratch) [+]</h4></summary>
  
##### <ins>Finish setup</ins> 
```sudo apt update
sudo apt dist-upgrade
sudo reboot

sudo apt autoremove
sudo apt install nvidia-jetpack
sudo nvpmodel -m 0 # Set power mode
sudo reboot
```
Check out the Jetson Orin series power modes: [Link](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency).

Next, Install the required packages and LLVM:
```
sudo apt-get -y install git python3-pip python3.10-dev nano autoconf bc build-essential clang
pip install --upgrade testresources setuptools wheel  
pip install numpy onnx --force-reinstall

BUILD_DIR=~/jax_build_dir
mkdir -p $BUILD_DIR
cd $BUILD_DIR
if [ ! -f llvm.sh ]; then
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
fi
sudo ./llvm.sh 18
sudo ln -sf /usr/bin/llvm-config-18 /usr/bin/llvm-config
sudo ln -sf /usr/bin/clang-18 /usr/bin/clang
```

It is recommended to increase the swap memory size of the Orin Nano by 8GB. Check out the * *Enlarge memory swap* * section [here](https://qengineering.eu/install-pytorch-on-jetson-nano.html).
  
##### <ins>Install Jax</ins>  

**Build Jax from Scratch (v0.4.26)**  
```
cd $BUILD_DIR
rm -rf jax
git clone --branch "jaxlib-v0.4.26" --depth=1 --recursive https://github.com/google/jax
cd jax

sudo python3 build/build.py \
    --enable_cuda \
    --cuda_compute_capabilities=sm_87 \
    --enable_nccl=False

## Install the built jaxlib wheel
pip install dist/jaxlib-0.4.26*.whl
```
##### Install Python packages
```
pip install onnx numpy==1.25.2 flax==0.8.1 orbax-checkpoint==0.4.3 "tensorstore==0.1.50" optax==0.2.2 chex==0.1.86 jax==0.4.26 gymnasium==0.29.0 seaborn==0.13.2 termcolor==2.4.0 tensorboardX==2.6.2.2 pyopengl==3.1.7 wandb==0.16.0 tensorflow_probability==0.21.0 imageio==2.34.1 dm_control==1.0.15 opencv_python==4.9.0.80 pyserial mujoco==3.1.6 jetson-stats
```

</details>

#### Local Server
```
# Clone the repo
git clone https://github.com/fahimfss/JSAC.git
cd JSAC

# Create a conda env and install JSAC
conda create -n jsac python=3.10
conda activate jsac

pip install -U "jax[cuda12]==0.6.2"
pip install -e .
```

## Training

Run a [task](https://github.com/fahimfss/JSAC/tree/master/tasks/simulation) file using:  
```python3 task_mujoco.py --seed 41```  

## Acknowledgements
- JSAC builds upon [Relod](https://github.com/rlai-lab/relod), a PyTorch-based RL training system used for robotic tasks at the [RLAI](http://rlai.ualberta.ca/) lab. 
- The JAX/Flax implementations were influenced by [jaxrl](https://github.com/ikostrikov/jaxrl).
- The Create2 environment is based on the implementation in [SenseAct](https://github.com/kindredresearch/SenseAct/tree/master/senseact/envs/create2). 

## Cite JSAC
```
@software{fahim_shahriar_jsac,
  author       = {Fahim Shahriar},
  title        = {fahimfss/JSAC: v1.0.0},
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15285797},
  url          = {https://doi.org/10.5281/zenodo.15285797},
}
```
