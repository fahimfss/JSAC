from collections import defaultdict
import json
import os
import shutil
import numpy as np
from termcolor import colored
from tensorboardX import SummaryWriter
from jsac.helpers.utils import show_learning_curve
from multiprocessing import Queue, Process
import jaxlib
import time


FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), 
            ('step', 'S', 'int'),
            ('episode_steps', 'ES', 'int'),
            ('duration', 'D', 'float'), 
            ('return', 'R', 'float'),
            ('actor_loss', 'AL', 'float'),
            ('entropy', 'ENT', 'float'),
            ('critic_loss', 'CL', 'float'), 
            ('num_updates', 'NU', 'int'),
            ('update_time', 'UT', 'float'),
            ('rb_sample_time', 'RBST', 'float'),
            ('action_sample_time', 'AST', 'float'), 
            ('env_time', 'ENVT', 'float'),
            ('elapsed_time', 'ELT', 'float'),
            ('battery_charge', 'BC', 'int')
        ],
        'eval': [('step', 'S', 'int'), ('return', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0
        self._value = 0
        
    def update_average(self, value, n=1):
        self._sum += value
        self._count += n
    
    def update_value(self, value):
        self._value = value

    def value(self):
        if self._value > 0:
            return self._value
        
        return self._sum / max(1, self._count)
        


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._value_items = ['num_updates', 
                             'battery_charge', 
                             'episode', 
                             'episode_steps', 
                             'duration', 
                             'return', 
                             'step', 
                             'elapsed_time']
        self._int_value_items = ['num_updates', 
                                 'battery_charge', 
                                 'episode', 
                                 'episode_steps', 
                                 'step']

    def log(self, key, value, n=1):
        if key in self._value_items:
           self._meters[key].update_value(value)
        else:
            self._meters[key].update_average(value, n)

    def _prime_meters(self, step, sw):
        data = dict()
        for key, meter in self._meters.items():
            tb_key = key
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            if key in self._int_value_items:
                value = int(meter.value())
            else:
                value = meter.value()
            data[key] = value
            if sw is not None:
                sw.add_scalar(tb_key, value, step)
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix, sw, wandb_log=False):
        if len(self._meters) == 0:
            return
        data = self._prime_meters(step, sw)
        data['step'] = step
        if wandb_log:
            wandb.log(data)
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, 
                 log_dir, 
                 xtick, 
                 args, 
                 use_tb=False, 
                 use_wandb=False, 
                 wandb_project_name='', 
                 wandb_run_name='', 
                 wandb_resume=False, 
                 config='rl'):
        
        self._log_queue = Queue()
        self._log_dir = log_dir
        self._xtick = xtick
        self._args = args
        self._use_tb = use_tb
        self._use_wandb = use_wandb
        self._wandb_project_name = wandb_project_name
        self._wandb_run_name = wandb_run_name
        self._wandb_resume = wandb_resume
        self._config = config

        self._log_process = Process(target=self._run)
        self._log_process.start()
        
    def push(self, data):
        self._log_queue.put(data)
    
    def plot(self):
        self._log_queue.put('plot')

    def _init(self):
        self._log_dir = self._log_dir
        if self._use_tb:
            tb_dir = os.path.join(self._log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(self._log_dir, 'train.log'),
            formating=FORMAT_CONFIG[self._config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(self._log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[self._config]['eval']
        )

        if self._use_wandb:
            import wandb
            self._use_wandb = True
            id = f'{self._wandb_project_name}-{self._wandb_run_name}'
            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_run_name,
                id=id,
                config=self._args,
                resume=self._wandb_resume
            )
        else:
            self._use_wandb = False

        self._returns=[]
        self._episode_steps=[]

        log_path = os.path.join(self._log_dir, 'train.log')
        if os.path.exists(log_path):
            with open(log_path, 'r') as ret_file:
                for line in ret_file.readlines():
                    dict = eval(line)
                    self._returns.append(dict['return'])
                    self._episode_steps.append(dict['episode_steps'])

        start_step = self._args['start_step']
        config_name = f'config_{start_step}.txt'
        config_path = os.path.join(self._log_dir, config_name) 
        with open(config_path, 'w') as cfl:
            for key, value in self._args.items():
                cfl.write(f'{key} -> {value}')
                cfl.write('\n\n')

        self._plot_path = os.path.join(self._log_dir, 'learning_curve.png') 

    def _run(self):
        self._init()

        while True:
            data = self._log_queue.get()

            if isinstance(data, str):
                if data == 'close':
                    return
                if data == 'plot':
                    try:
                        self._plot_returns()
                    except Exception as e:
                        print(e)
                        exit(0)

                    continue

            step = data['step']
            tag = data['tag']
            for k, v in data.items():
                if k in ['tag', 'dump', 'step', 'TimeLimit.truncated']:
                    continue
                elif k == 'return':
                    self._returns.append(v)
                elif k == 'episode_steps':
                    self._episode_steps.append(v)

                if isinstance(v, jaxlib.xla_extension.ArrayImpl):
                    v = v.item()
                self._log(f'{tag}/{k}', v, step)

            if data['dump'] == True:
                self._dump(step)

 
    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def _log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        # self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def _dump(self, step):
        self._train_mg.dump(step, 'train', self._sw, self._use_wandb)
        self._eval_mg.dump(step, 'eval', self._sw, self._use_wandb)

    def _plot_returns(self):
        show_learning_curve(self._plot_path, 
                            self._returns, 
                            self._episode_steps, 
                            self._xtick)

    def close(self):
        self._log_queue.put('close')
        time.sleep(2)
        if self._use_wandb:
            wandb.finish()

    

