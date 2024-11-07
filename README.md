# JSAC

## Installation
<details>
<summary><h3>Orin (from scratch)</h3></summary>
  
#### <ins>Finish setup</ins> 
```sudo apt update
sudo apt dist-upgrade
sudo reboot
sudo apt autoremove
sudo apt install nvidia-jetpack
```

Install the required packages
```
sudo apt -y install python3-pip
pip install --upgrade testresources setuptools wheel  
sudo apt-get -y install autoconf bc build-essential g++-10 gcc-10 clang-8 python3.8-dev
pip install numpy onnx --force-reinstall
```  

  
#### <ins>Install Jax</ins>  

**Option 1: Build Jax from Scratch (v0.4.12)**  
```
git clone -b jax-v0.4.12 https://github.com/google/jax
cd jax
python3 build/build.py --enable_cuda --cuda_compute_capabilities=sm_87
## Install the built jaxlib wheel
pip install -e .
```

**Option 2: Build Jax using wheel (v0.4.12, Orin JetPack 5.1.2)**  
Download the Jaxlib wheel from [here](https://drive.google.com/file/d/1UBxzqAxperW-4m44G1htKkAVEA6LfZlT/view?usp=drive_link) and install Jaxlib.  
Install Jax using ``` pip install jax==0.4.12 ```  
  
  
#### <ins>Update gcc, g++, and install the remaining libraries</ins>
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 2
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 2
sudo update-alternatives --config g++
```

```
pip install flax==0.7.2
pip install optax==0.1.7
pip install gym==0.23.1 matplotlib==3.7.2 tensorboardX==2.6.2 termcolor==2.3.0
pip install mujoco-py wandb seaborn pandas==1.5.3
```

</details>

### Local Server
```
# Clone the repo
git clone https://github.com/fahimfss/JSAC.git
cd JSAC

# Create a conda env and install JSAC
conda create -n jsac python=3.10
conda activate jsac

conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3

pip install -U "jax[cuda12]==0.4.30"
pip install -e .
```
Installing the latest Nvidia driver might be required to run JAX properly

<br>  

### Training

Run a [task](https://github.com/fahimfss/JSAC/tree/master/tasks/simulation) file using:  
```python3 task_mujoco.py --seed 41```  

If EGL fails, use prefix: `PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa`
