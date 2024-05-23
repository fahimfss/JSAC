import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
from jsac.helpers.logger import Logger
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
import time 
import argparse
import multiprocessing as mp
import shutil
import numpy as np


config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [256, 256],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--name', default='hopper_hop_sync_prop', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='hopper_hop', type=str)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=10000, type=int)
    parser.add_argument('--env_steps', default=50000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=True, action='store_true')
    parser.add_argument('--calculate_grad_norm', default=True, action='store_true')
    
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--clip_global_norm', default=0.5, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)

    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    
    # vision parameters, used when mode is 'img' or 'img_prop'
    parser.add_argument('--image_height', default=84, type=int)
    parser.add_argument('--image_width', default=84, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--spatial_softmax', default=True, action='store_true')
    parser.add_argument('--rad_offset', default=0.01, type=float)
    
    # misc
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, action='store_true')
    parser.add_argument('--xtick', default=10000, type=int)
    parser.add_argument('--plot_every', default=50000, type=int)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=-1, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args

def main(seed=-1):
    args = parse_args()
 
    args.apply_rad = False
    args.spatial_softmax = False
    args.use_critic_encoder = False

    if seed != -1:
        args.seed = seed

    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
        elif inp == '':
            pass
        else:
            exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = f'{args.work_dir}/checkpoints/'

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    args.net_params = config

    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb, wandb_project_name, 
                   wandb_run_name, args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)

    env = DMCVisualEnv(args.env_name, 
                       args.mode, 
                       args.seed,
                       args.image_history, 
                       args.image_width, 
                       args.image_height, 
                       args.num_cameras)
    
    env = WrappedEnv(env, start_step=args.start_step, 
                     start_episode=args.start_episode)

    set_seed_everywhere(seed=args.seed)

    args.image_shape = env.image_space.shape
    args.single_image_shape = (args.image_height, args.image_width, 3*args.num_cameras)
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    
    if args.sync_mode:
        agent = SACRADAgent(vars(args))
    else:
        agent = AsyncSACRADAgent(vars(args))

    task_start_time = time.time()
    proprioception = env.reset()
    first_step = True

    while env.total_steps < args.env_steps:
        t1 = time.time()
        action = agent.sample_actions(proprioception)
        t2 = time.time()
        next_proprioception, reward, done, info = env.step(action)
        t3 = time.time()

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        agent.add(proprioception, action, reward, next_proprioception, mask, first_step)
        first_step = False

        proprioception = next_proprioception

        if env.total_steps > args.init_steps:
            update_infos = agent.update()
            if update_infos is not None and env.total_steps %  args.log_every == 0:
                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False
                    L.push(update_info)
                    
        if done:
            proprioception = env.reset()
            first_step = True
            info['tag'] = 'train'
            info['dump'] = True
            info['elapsed_time'] = time.time() - task_start_time
            L.push(info)

        if env.total_steps % args.plot_every == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)

    if args.save_model:
        agent.checkpoint(env.total_steps)
    # L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # for i in range(30):
    #     main(i)

    main()