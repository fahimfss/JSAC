import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere
from jsac.helpers.logger import Logger
from jsac.envs.create2_orin_visual_reacher.env import Create2VisualReacherEnv
from jsac.helpers.utils import WrappedEnv
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from threading import Thread
import time
from tensorboardX import SummaryWriter
import tqdm
import argparse
import shutil
import multiprocessing as mp
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

    'mlp': [512, 512],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--name', default='create2_orin_visual_back_reacher', type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--image_width', default=120, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)

    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--episode_length_time', default=15.0, type=float)
    parser.add_argument('--dt', default=0.045, type=float)
    parser.add_argument('--min_target_size', default=0.2, type=float)
    parser.add_argument('--reset_penalty_steps', default=67, type=int)
    parser.add_argument('--min_charge', default=810, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    parser.add_argument('--pause_before_reset', default=0, type=float)
    parser.add_argument('--pause_after_reset', default=0, type=float)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=150000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=175000, type=int)
    parser.add_argument('--task_timeout_mins', default=95, type=int)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--apply_rad', default=True, action='store_true')
    parser.add_argument('--rad_offset', default=0.01, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--use_critic_encoder', default=True, 
                        action='store_true')
    
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--spatial_softmax', default=True, action='store_true')
    
    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    
    # misc
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=1000, type=int)
    parser.add_argument('--save_wandb', default=True, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=5000, type=int)
    parser.add_argument('--load_model', default=101770, type=int)
    parser.add_argument('--start_step', default=101771, type=int)
    parser.add_argument('--start_episode', default=151, type=int)

    parser.add_argument('--buffer_save_path', default='./buffers/', type=str)
    parser.add_argument('--buffer_load_path', default='./buffers/', type=str)

    args = parser.parse_args()
    return args

def get_run_flag():
    with open('run_flags.txt', 'r') as f:
        return int(f.readline())

def main(seed=-1):
    args = parse_args()

    RF_CONTINUE = 0
    RF_END_RUN_WO_SAVE = 1
    RF_END_RUN_W_SAVE = 2

    assert args.mode == MODE.IMG_PROP
    assert args.sync_mode == False

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
            print('Previous work dir removed.')
        elif inp == '':
            pass
        else:
            exit(0)

    rf = get_run_flag()
    if rf != RF_CONTINUE:
        print('Ending the run as the RF flag is set to END_RUN in', end=' ')
        print('run_flags.txt.\nPlease set the flag to 0 and try again!')
        exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        make_dir(args.buffer_save_path)

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    args.net_params = config

    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, vars(args), args.save_tensorboard, 
                   args.save_wandb, wandb_project_name, wandb_run_name, 
                   args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, vars(args), args.save_tensorboard, 
                   args.save_wandb)

    image_shape = (args.image_height, args.image_width, 3*args.stack_frames)

    env = Create2VisualReacherEnv(
        episode_length_time=args.episode_length_time, 
        dt=args.dt,
        image_shape=image_shape,
        camera_id=args.camera_id,
        min_target_size=args.min_target_size,
        pause_before_reset=args.pause_before_reset,
        pause_after_reset=args.pause_after_reset)
    
    episode_length_step = int(args.episode_length_time / args.dt)
    env = WrappedEnv(env, 
                     episode_max_steps=episode_length_step,
                     is_min_time=True,
                     reward_penalty=args.reset_penalty_steps * args.reward,
                     steps_penalty=args.reset_penalty_steps,
                     start_step = args.start_step,
                     start_episode = args.start_episode)
    
    set_seed_everywhere(seed=args.seed)
    env.start()

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    action_dim = args.action_shape[-1]
    args.env_action_space = env.action_space

    if args.sync_mode:
        agent = SACRADAgent(args)
    else:
        agent = AsyncSACRADAgent(args)

    task_start_time = time.time()
    task_end_time = task_start_time + (args.task_timeout_mins * 60)
    update_paused = True
    (image, proprioception) = env.reset()

    while env.total_steps <= args.env_steps:
        t1 = time.time()
        action = agent.sample_actions((image, proprioception))
        t2 = time.time()
        (next_image, next_proprioception), reward, done, info = env.step(action)
        t3 = time.time()
        
        mask = 0.0 if done else 1.0
        agent.add((image, proprioception), action, reward, 
                  (next_image, next_proprioception),  mask)
        image = next_image
        proprioception = next_proprioception

        if done or 'TimeLimit.truncated' in info:
            charge = info['battery_charge']
            elapsed_time = "{:.3f}".format(time.time() - task_start_time)

            if done:
                info['tag'] = 'train'
                info['elapsed_time'] = time.time() - task_start_time
                info['dump'] = True
                L.push(info)
                (image, proprioception) = env.reset()
            else:
                episode = info['episode']
                sub_epi = info['sub_episode']
                print(f'>> Episode {episode}, sub-episode {sub_epi} done. ' + 
                  f'Step: {env.total_steps}, Elapsed time: {elapsed_time}s')
                (image, proprioception) = env.reset(reset_stats=False)

            rf = get_run_flag()
            if rf == RF_END_RUN_WO_SAVE or rf == RF_END_RUN_W_SAVE:
                break
                
            if time.time() > task_end_time or charge < args.min_charge:
                rf = RF_END_RUN_W_SAVE
                break

            if update_paused and env.total_steps >= args.init_steps \
                and charge > args.min_charge:
                agent.resume_update()
                update_paused = False
                time.sleep(20)

        if not update_paused and env.total_steps >= args.init_steps:
            update_infos = agent.update()
            if update_infos is not None:
                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False
                    L.push(update_info)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)
    
    agent.pause_update()
    env.close()

    if rf != RF_END_RUN_WO_SAVE and args.save_model:
        agent.checkpoint(env.total_steps)
    L.plot()
    L.close()

    if rf == RF_END_RUN_WO_SAVE:
        agent.close(without_save=True)
    else:
        agent.close()
    

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

