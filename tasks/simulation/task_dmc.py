import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import argparse
import multiprocessing as mp

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

from jsac.helpers.logger import Logger
from jsac.helpers.eval import start_eval_process
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv


config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 5, 2],
        [32, 32, 5, 2],
        [32, 64, 3, 1],
        [64, 64, 3, 1],
    ],
    
    'latent_dim': 64,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='cheetah', type=str)
    parser.add_argument('--image_height', default=96, type=int)     # Mode: img, img_prop
    parser.add_argument('--image_width', default=96, type=int)      # Mode: img, img_prop     
    parser.add_argument('--image_history', default=3, type=int)     # Mode: img, img_prop
    parser.add_argument('--action_repeat', default=2, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=500_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=500_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=True, action='store_true')
    parser.add_argument('--global_norm', default=5.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float) 
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)

    # NTK
    parser.add_argument('--ntk', default=0.0, type=float)
    
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)   # Sync mode: False
    
    # encoder
    parser.add_argument('--spatial_softmax', default=False, action='store_true')    # Mode: img, img_prop

    # sac
    parser.add_argument('--temp_lr', default=3e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=10_000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=500_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args

def main(seed=-1):
    task_start_time = time.time()
    args = parse_args()

    if seed != -1:
        args.seed = seed

    if not args.sync_mode:
        assert args.mode != MODE.PROP, "Async mode is not supported for proprioception only tasks." 

    sync_mode = 'sync' if args.sync_mode else 'async'
    args.name = f'{args.env_name}_{args.mode}_{sync_mode}'

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

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    if args.save_model:
        make_dir(args.model_dir)
        
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
                       args.num_cameras,
                       args.action_repeat)
    
    env = WrappedEnv(env, start_step=args.start_step, 
                     start_episode=args.start_episode)

    set_seed_everywhere(seed=args.seed)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space

    if args.sync_mode:
        agent = SACRADAgent(vars(args)) 
    else:
        agent = AsyncSACRADAgent(vars(args))
        
    if args.eval_steps > 0:
        eval_args = vars(args)
        eval_args['env_type'] = 'DMC'
        eval_queue = mp.Queue()        
        eval_process = start_eval_process(eval_args, 
                                          args.work_dir, 
                                          eval_queue, 
                                          args.num_eval_episodes)

    update_paused = True
    state = env.reset()
    first_step = True

    while env.total_steps < args.env_steps:
        t1 = time.time()
        if env.total_steps < args.init_steps + 100:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(state)
        t2 = time.time()
        next_state, reward, done, info = env.step(action)
        t3 = time.time()

        mask = 1.0 if not done or 'truncated' in info else 0.0
        
        agent.add(state, action, reward, next_state, mask, first_step)
        first_step = False
        state = next_state

        if done or 'truncated' in info:
            state = env.reset()
            first_step = True
            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - task_start_time
            info['dump'] = True
            L.push(info)

        if env.total_steps > args.init_steps and env.total_steps % args.update_every == 0:
            if not args.sync_mode and update_paused: 
                agent.resume_update()
                update_paused = False
            update_infos = agent.update()
            if update_infos is not None and env.total_steps % args.log_every == 0:
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
            
        if env.total_steps % args.eval_steps == 0:
            eval_queue.put(agent.get_actor_params())
            eval_queue.put(env.total_steps)

    if not args.sync_mode:
        agent.pause_update()
    if args.save_model:
        agent.checkpoint(env.total_steps)
        
    if args.eval_steps > 0:    
        eval_queue.put('close')
        eval_process.join()
        
    L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

