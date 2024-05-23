import numpy as np
import collections
import threading
import time
import os
import pickle
from multiprocessing import shared_memory, Process, Queue, Lock


SB_0 = 'sb0'
SB_1 = 'sb1'
READY = 'ready'
START = 'start'
SAVE = 'save'
CLOSE = 'close'

## rb: replay buffer
## sb: sampling buffer
## sp: sampling process
## sm: shared memory
## ip: inference process

Batch = collections.namedtuple(
    'Batch', ['images', 'proprioceptions', 'actions', 'rewards',
              'dones', 'next_images', 'next_proprioceptions'])


class ReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, 
                 image_shape, 
                 proprioception_shape, 
                 action_shape,
                 capacity, 
                 batch_size, 
                 init_buffers=True, 
                 load_path=''):

        self._image_shape = image_shape
        self._proprioception_shape = proprioception_shape
        self._action_shape = action_shape
        self._capacity = capacity
        self._batch_size = batch_size

        self._idx = 0
        self._full = False
        self._count = 0
        self._steps = 0
        self._lock = Lock()

        self._ignore_image = True
        self._ignore_propri = True

        if image_shape is not None:
            self._ignore_image = False

        if proprioception_shape is not None:
            self._ignore_propri = False

        if load_path:
            self._load_path = load_path
        else:
            self._load_path = ''

        if init_buffers:
            self._init_buffers()

    def _init_buffers(self):
        total_size = 0

        if self._load_path:
            total_size = self._load()
        else:
            if not self._ignore_image:
                self._images = np.empty(
                    (self._capacity, *self._image_shape), 
                    dtype=np.uint8)
                self._next_images = np.empty(
                    (self._capacity, *self._image_shape), 
                    dtype=np.uint8)
                
                total_size += self._images.nbytes + self._next_images.nbytes

            if not self._ignore_propri:
                self._propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                self._next_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                
                total_size += self._propris.nbytes + self._next_propris.nbytes

            self._actions = np.empty(
                (self._capacity, *self._action_shape), 
                dtype=np.float32)
            
            self._rewards = np.empty((self._capacity), 
                                     dtype=np.float32)
            self._dones = np.empty((self._capacity), 
                                   dtype=np.float32)
            
            total_size += self._actions.nbytes + self._rewards.nbytes + self._dones.nbytes
        
        return total_size

    def add(self, 
            image, 
            propri, 
            action, 
            reward, 
            next_image, 
            next_propri, 
            done):
        if not self._ignore_image:
            self._images[self._idx] = image
            self._next_images[self._idx] = next_image
        if not self._ignore_propri:
            self._propris[self._idx] = propri
            self._next_propris[self._idx] = next_propri
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = done

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0
        self._count = self._capacity if self._full else self._idx
        self._steps += 1

    def sample(self):
        idxs = np.random.randint(0, 
                                 self._count,
                                 size=min(self._count, self._batch_size))
        
        if self._ignore_image:
            images = None
            next_images = None
        else:
            images = self._images[idxs]
            next_images = self._next_images[idxs]

        if self._ignore_propri:
            propris = None
            next_propris = None
        else:
            propris = self._propris[idxs]
            next_propris = self._next_propris[idxs]

        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        dones = self._dones[idxs]

        return Batch(images=images, 
                     proprioceptions=propris,
                     actions=actions, 
                     rewards=rewards, 
                     dones=dones,
                     next_images=next_images, 
                     next_proprioceptions=next_propris)


    def save(self, save_path):
        tic = time.time()
        print(f'Saving the replay buffer in {save_path}..')
        with self._lock:
            data = {
                'count': self._count,
                'idx': self._idx,
                'full': self._full,
                'steps': self._steps
            }

            with open(os.path.join(save_path, "buffer_data.pkl"),
                      "wb") as handle:
                pickle.dump(data, handle, protocol=4)

            if not self._ignore_image:
                np.save(os.path.join(save_path, "images.npy"), self._images)
                np.save(os.path.join(save_path, "next_images.npy"),
                        self._next_images)

            if not self._ignore_propri:
                np.save(os.path.join(save_path, "propris.npy"), self._propris)
                np.save(os.path.join(save_path, "next_propris.npy"),
                        self._next_propris)

            np.save(os.path.join(save_path, "actions.npy"), self._actions)
            np.save(os.path.join(save_path, "rewards.npy"), self._rewards)
            np.save(os.path.join(save_path, "dones.npy"), self._dones)

        print("Saved the buffer locally,", end=' ')
        print("took: {:.3f}s.".format(time.time() - tic))

    def _load(self):
        tic = time.time()
        print("Loading buffer")

        data = pickle.load(open(os.path.join(self._load_path,
                                             "buffer_data.pkl"), "rb"))
        self._count = data['count']
        self._idx = data['idx']
        self._full = data['full']
        self._steps = data['steps']

        if not self._ignore_image:
            self._images = np.load(os.path.join(self._load_path, "images.npy"))
            self._next_images = np.load(os.path.join(self._load_path,
                                                     "next_images.npy"))

        if not self._ignore_propri:
            self._propris = np.load(os.path.join(self._load_path,
                                                 "propris.npy"))
            self._next_propris = np.load(os.path.join(self._load_path,
                                                      "next_propris.npy"))

        self._actions = np.load(os.path.join(self._load_path, "actions.npy"))
        self._rewards = np.load(os.path.join(self._load_path, "rewards.npy"))
        self._dones = np.load(os.path.join(self._load_path, "dones.npy"))

        print("Loaded the buffer from: {}".format(self._load_path), end=' ')
        print("Took: {:.3f}s".format(time.time() - tic))
        
    def close(self):
        pass


class AsyncSMReplayBuffer(ReplayBuffer):
    def __init__(self, 
                 image_shape, 
                 proprioception_shape, 
                 action_shape, 
                 capacity, 
                 batch_size, 
                 obs_queue, 
                 load_path=''):
        
        super().__init__(
            image_shape, 
            proprioception_shape, 
            action_shape, 
            capacity,
            batch_size, 
            False, 
            load_path)
        
        sizes = self._get_sb_sizes(batch_size)

        self._obs_queue = obs_queue

        self._rcv_from_sampling_process_queue = Queue()
        self._send_to_sampling_process_queue = Queue()

        self._start_batch = False

        self._producer_process = Process(target=self._produce_samples_sp)
        self._producer_process.start()

        self._sb_0, self._sb_0_sm, sb_0_sm_names = self._create_sm_sb(sizes)
        self._sb_1, self._sb_1_sm, sb_1_sm_names = self._create_sm_sb(sizes)

        self._send_to_sampling_process_queue.put(sb_0_sm_names)
        self._send_to_sampling_process_queue.put(sb_1_sm_names)

        self._last_sb = None


    def sample(self):
        if not self._start_batch:
            self._start_batch = True
            self._obs_queue.put('start')
        sb_code = self._rcv_from_sampling_process_queue.get()

        if self._last_sb is not None:
            self._send_to_sampling_process_queue.put(self._last_sb)

        self._last_sb = sb_code

        if sb_code == SB_0:
            batch = self._sb_0
        else:
            batch = self._sb_1

        return batch
    
    def _recv_obs_sp(self):
        while True:
            observation = self._obs_queue.get()
            if isinstance(observation, str):
                if observation == CLOSE:
                    return
                if observation == START:
                    self._start_batch = True
                    continue

            with self._lock:
                self.add(*observation)

    def _get_sb_sizes(self, batch_size):
        image_size = 0
        proprioception_size = 0
        if not self._ignore_image:
            image_size = np.random.randint(
                0, 256, size=(batch_size, *self._image_shape), 
                dtype=np.uint8).nbytes
        if not self._ignore_propri:
            proprioception_size = np.random.uniform(
                size=(batch_size, *self._proprioception_shape)
                ).astype(np.float32).nbytes

        action_size = np.random.uniform(
            size=(batch_size, *self._action_shape)).astype(np.float32).nbytes
        
        done_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes
        
        reward_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes

        return {
            'img_sb_size': image_size,
            'proprioception_sb_size': proprioception_size, 
            'action_sb_size': action_size,
            'done_sb_size': done_size,
            'reward_sb_size': reward_size
            }

    def _create_sm_sb(self, sizes):        
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(create=True, size=sizes['img_sb_size'])
            next_img_sm = shared_memory.SharedMemory(create=True, 
                                                      size=sizes['img_sb_size'])
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_sm.buf)

        propris = None
        next_propris = None
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            next_proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)


        action_sm = shared_memory.SharedMemory(create=True, size=sizes['action_sb_size'])
        done_sm = shared_memory.SharedMemory(create=True, size=sizes['done_sb_size'])
        reward_sm = shared_memory.SharedMemory(create=True, size=sizes['reward_sb_size'])

        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_sm.buf)
        dones = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=done_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_sm.buf)
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, dones=dones,
                      next_images=next_images, next_proprioceptions=next_propris)
        
        sm_names = {}
        if not self._ignore_image:
            sm_names['img_sm'] = img_sm.name
            sm_names['next_img_sm'] = next_img_sm.name
        if not self._ignore_propri:
            sm_names['proprioception_sm'] = proprioception_sm.name
            sm_names['next_proprioception_sm'] = next_proprioception_sm.name
        sm_names['action_sm'] = action_sm.name
        sm_names['done_sm'] = done_sm.name
        sm_names['reward_sm'] = reward_sm.name
        
        sms = (img_sm, next_img_sm,  proprioception_sm, 
                next_proprioception_sm, action_sm, done_sm, reward_sm)

        return sb, sms, sm_names
    
    def _copy_batch(self, batch_src, batch_dest):
        if not self._ignore_image:
            np.copyto(batch_dest.images, batch_src.images)
            np.copyto(batch_dest.next_images, batch_src.next_images)

        if not self._ignore_propri:
            np.copyto(batch_dest.proprioceptions, batch_src.proprioceptions)
            np.copyto(batch_dest.next_proprioceptions, 
                      batch_src.next_proprioceptions)

        np.copyto(batch_dest.actions, batch_src.actions)
        np.copyto(batch_dest.rewards, batch_src.rewards)
        np.copyto(batch_dest.dones, batch_src.dones)

    def _get_sm_sb_sp(self, sm_names):
        total_size = 0
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(
                name=sm_names['img_sm'])
            next_img_sm = shared_memory.SharedMemory(
                name=sm_names['next_img_sm'])
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_sm.buf)
            total_size += images.nbytes + next_images.nbytes

        propris = None
        next_propris = None    
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(
                name=sm_names['proprioception_sm'])
            next_proprioception_sm = shared_memory.SharedMemory(
                name=sm_names['next_proprioception_sm'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)
            
            total_size += propris.nbytes + next_propris.nbytes

        action_sm = shared_memory.SharedMemory(
            name=sm_names['action_sm'])
        done_sm = shared_memory.SharedMemory(
            name=sm_names['done_sm'])
        reward_sm = shared_memory.SharedMemory(
            name=sm_names['reward_sm'])
        
        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_sm.buf)
        dones = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=done_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_sm.buf)
        
        total_size += actions.nbytes + dones.nbytes + rewards.nbytes
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, dones=dones,
                      next_images=next_images, next_proprioceptions=next_propris)
              
        sms = (img_sm, next_img_sm,  proprioception_sm, 
                next_proprioception_sm, action_sm, done_sm, reward_sm)

        return sb, sms, total_size

    def _produce_samples_sp(self): 
        rb_size = self._init_buffers()
        self._recv_obs_thread = threading.Thread(target=self._recv_obs_sp)
        self._recv_obs_thread.start()

        self._rcv_from_ip_queue = self._send_to_sampling_process_queue
        self._send_to_ip_queue = self._rcv_from_sampling_process_queue

        sb_0_sm_names = self._rcv_from_ip_queue.get()
        sb_1_sm_names = self._rcv_from_ip_queue.get()

        self._sb_0, self._sb_0_sm, sb_0_size = self._get_sm_sb_sp(sb_0_sm_names)
        self._sb_1, self._sb_1_sm, sb_1_size = self._get_sm_sb_sp(sb_1_sm_names)
        
        total_size = rb_size + sb_0_size + sb_1_size
        print(f'Total size of buffers (in GB): {total_size / 1e9}')

        while not self._start_batch:
            # Checking if the replay buffer process 
            # needs to be closed while waiting
            if not self._rcv_from_ip_queue.empty():
                code = self._rcv_from_ip_queue.get()
                if code == CLOSE:
                    self._close_sp()
                    return
                elif code == SAVE:
                    self._rcv_from_ip_queue.get()
            time.sleep(0.1)

        with self._lock:
            sb_0 = super().sample()
        self._copy_batch(sb_0, self._sb_0)

        with self._lock:
            sb_1 = super().sample()
        self._copy_batch(sb_1, self._sb_1)

        self._send_to_ip_queue.put(SB_0)
        self._send_to_ip_queue.put(SB_1)

        while True:
            with self._lock:
                batch = super().sample()
            code = self._rcv_from_ip_queue.get()
            if code == SB_0:
                self._copy_batch(batch, self._sb_0)
                self._send_to_ip_queue.put(SB_0)
            elif code == SB_1:
                self._copy_batch(batch, self._sb_1)
                self._send_to_ip_queue.put(SB_1)
            elif code == SAVE:
                save_path = self._rcv_from_ip_queue.get()
                super().save(save_path)
            elif code == CLOSE:
                break

        self._close_sp()

    def _close_sp(self):
        self._obs_queue.put('close')
        print('Closng replay buffer shared memory..')
        with self._lock:
            for mem in self._sb_0_sm:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass
            for mem in self._sb_1_sm:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass
        
    def save(self, save_path):
        self._send_to_sampling_process_queue.put(SAVE)
        self._send_to_sampling_process_queue.put(save_path)
        time.sleep(1)
        self._lock.acquire()
        self._lock.release()

    def close(self):
        self._send_to_sampling_process_queue.put(CLOSE)
        self._producer_process.join()

        for mem in self._sb_0_sm:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass
        for mem in self._sb_1_sm:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass


class AsyncSampleEfficientReplayBuffer():

    def __init__(self, 
                 image_shape,
                 proprioception_shape, 
                 action_shape,
                 capacity,
                 batch_size,
                 obs_queue,
                 num_workers=8,
                 max_num_episodes=-1,
                 max_episode_lenght=10000,
                 image_history=3,
                 image_type='hwc',
                 load_path=''):

        self._image_shape = image_shape 
        self._proprioception_shape = proprioception_shape
        self._action_shape = action_shape
        self._capacity = capacity
        self._batch_size = batch_size
        self._num_workers = num_workers
        if max_num_episodes == -1:
            self._max_num_episodes = int(capacity/5)
        else:
            self._max_num_episodes = max_num_episodes
        self._max_episode_lenght = max_episode_lenght
        self._image_history = image_history
        self._image_type = image_type
        self._load_path = load_path

        self._worker_batch_size = batch_size // num_workers

        self._lock = Lock()
        self._obs_queue = obs_queue
        
        self._ignore_image = True
        self._ignore_propri = True

        if self._image_shape:
            self._ignore_image = False 
            if image_type == 'hwc':
                self._height, self._width, self._channels = self._image_shape
                self._stacked_image_shape = (self._height, self._width, self._channels * self._image_history)
            else:
                self._channels, self._height, self._width = self._image_shape
                self._stacked_image_shape = (self._channels * self._image_history, self._height, self._width)

        if proprioception_shape:
            self._ignore_propri = False

        if load_path:
            self._load_path = load_path
        else:
            self._load_path = ''

        self._rcv_from_sampling_process_queue = Queue()
        self._send_to_sampling_process_queue = Queue()

        self._start_batch = False

        self._prepare_async()

    def _prepare_async(self):
        sizes = self._get_sb_sizes(self._batch_size)

        self._sampling_process = Process(target=self._produce_samples_sp)
        self._sampling_process.start()

        self._sb_0, self._sb_0_sm, sb_0_sm_names = self._create_sm_sb(sizes)
        self._sb_1, self._sb_1_sm, sb_1_sm_names = self._create_sm_sb(sizes)

        self._send_to_sampling_process_queue.put(sb_0_sm_names)
        self._send_to_sampling_process_queue.put(sb_1_sm_names)

        self._last_sb = -1

        assert self._rcv_from_sampling_process_queue.get() == READY
        
    
    def sample(self):
        if not self._start_batch:
            self._start_batch = True
            self._obs_queue.put(START)
        sb_code = self._rcv_from_sampling_process_queue.get()

        if self._last_sb != -1:
            self._send_to_sampling_process_queue.put(self._last_sb)

        self._last_sb = sb_code

        if sb_code == SB_0:
            batch = self._sb_0
        else:
            batch = self._sb_1

        return batch

    def save(self, save_path):
        self._send_to_sampling_process_queue.put(SAVE)
        self._send_to_sampling_process_queue.put(save_path)
        time.sleep(1)
        self._lock.acquire()
        self._lock.release()

    def close(self):
        self._send_to_sampling_process_queue.put(CLOSE)
        self._sampling_process.join()
        for sm in self._sb_0_sm:
            if sm is not None:
                try:
                    sm.unlink()
                except:
                    pass
        for sm in self._sb_1_sm:
            if sm is not None:
                try:
                    sm.unlink()
                except:
                    pass

    def _save_sp(self, save_path):
        tic = time.time()
        print(f'Saving the replay buffer in {save_path}..')
        with self._lock:
            data = {
                'idx': self._idx,
                'image_idx': self._image_idx,
                'full': self._full,
                'count': self._count 
            }

            with open(os.path.join(save_path, "buffer_data.pkl"),
                      "wb") as handle:
                pickle.dump(data, handle, protocol=4)

            if not self._ignore_image:
                np.save(os.path.join(save_path, "rb_images.npy"), self._rb_images)
                np.save(os.path.join(save_path, "rb_image_references.npy"), self._rb_image_references)
                np.save(os.path.join(save_path, "rb_next_image_references.npy"), self._rb_next_image_references)
                np.save(os.path.join(save_path, "last_image_ref.npy"), self._last_image_ref)

            if not self._ignore_propri:
                np.save(os.path.join(save_path, "rb_propris.npy"), self._rb_propris)
                np.save(os.path.join(save_path, "rb_next_propris.npy"), self._rb_next_propris)

            np.save(os.path.join(save_path, "rb_actions.npy"), self._rb_actions)
            np.save(os.path.join(save_path, "rb_rewards.npy"), self._rb_rewards)
            np.save(os.path.join(save_path, "rb_dones.npy"), self._rb_dones)

        print("Saved the buffer locally,", end=' ')
        print("took: {:.3f}s.".format(time.time() - tic))

    def _load_sp(self):
        tic = time.time()

        total_size = 0

        print("Loading buffer")

        data = pickle.load(open(os.path.join(self._load_path,
                                             "buffer_data.pkl"), "rb"))
        self._count = data['count']
        self._idx = data['idx']
        self._full = data['full']
        self._image_idx = data['image_idx'] 

        if not self._ignore_image:
            images = np.load(os.path.join(self._load_path, "rb_images.npy")) 

            images_size = self._height * self._width * \
                self._image_capacity * np.dtype(np.uint8).itemsize
            self._rb_images_sm = shared_memory.SharedMemory(create=True, size=images_size)
            
            if self._image_type == 'hwc':    
                self._rb_images = np.ndarray(
                    (self._height, self._width, self._image_capacity), 
                    dtype=np.uint8, buffer=self._rb_images_sm.buf)
            else:
                self._rb_images = np.ndarray(
                    (self._image_capacity, self._height, self._width), 
                    dtype=np.uint8, buffer=self._rb_images_sm.buf)
                
            np.copyto(self._rb_images, images)

            self._rb_image_references = np.load(os.path.join(self._load_path, "rb_image_references.npy")) 
            self._rb_next_image_references = np.load(os.path.join(self._load_path, "rb_next_image_references.npy")) 
            self._last_image_ref  = np.load(os.path.join(self._load_path, "last_image_ref.npy")) 

            total_size += self._rb_images.nbytes + self._rb_image_references.nbytes + \
                self._rb_next_image_references.nbytes + self._last_image_ref.nbytes

        if not self._ignore_propri:
            self._rb_propris = np.load(os.path.join(self._load_path, "rb_propris.npy"))
            self._rb_next_propris = np.load(os.path.join(self._load_path, "rb_next_propris.npy"))

            total_size += self._rb_propris.nbytes + self._rb_next_propris.nbytes

        self._rb_actions = np.load(os.path.join(self._load_path, "rb_actions.npy"))
        self._rb_rewards = np.load(os.path.join(self._load_path, "rb_rewards.npy"))
        self._rb_dones = np.load(os.path.join(self._load_path, "rb_dones.npy"))

        total_size += self._rb_actions.nbytes + self._rb_rewards.nbytes + self._rb_dones.nbytes

        print("Loaded the buffer from: {}".format(self._load_path), end=' ')
        print("Took: {:.3f}s".format(time.time() - tic))

        return total_size

    def _recv_obs_sp(self):
        while True:
            observation = self._obs_queue.get()
            if isinstance(observation, str):
                if observation == CLOSE:
                    return
                if observation == START:
                    self._start_batch = True
                    continue

            with self._lock:
                self._add_sp(*observation)

    def _get_sb_sizes(self, batch_size):
        image_size = 0
        proprioception_size = 0
        if not self._ignore_image:
            image_size = np.dtype(np.uint8).itemsize * batch_size * np.prod(self._stacked_image_shape)
        
        if not self._ignore_propri:
            proprioception_size = np.dtype(np.float32).itemsize * batch_size * np.prod(self._proprioception_shape)

        action_size = np.dtype(np.float32).itemsize * batch_size * np.prod(self._action_shape)
        done_size = np.dtype(np.float32).itemsize * batch_size
        reward_size = np.dtype(np.float32).itemsize * batch_size

        return {
            'img_sb_size': image_size,
            'proprioception_sb_size': proprioception_size, 
            'action_sb_size': action_size,
            'done_sb_size': done_size,
            'reward_sb_size': reward_size
            }

    def _create_sm_sb(self, sizes):        
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(create=True, size=sizes['img_sb_size'])
            next_img_sm = shared_memory.SharedMemory(create=True, size=sizes['img_sb_size'])
            
            images = np.ndarray(
                (self._batch_size, *self._stacked_image_shape), dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray(
                (self._batch_size, *self._stacked_image_shape), dtype=np.uint8, buffer=next_img_sm.buf)

        propris = None
        next_propris = None
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            next_proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)

        action_sm = shared_memory.SharedMemory(create=True, size=sizes['action_sb_size'])
        done_sm = shared_memory.SharedMemory(create=True, size=sizes['done_sb_size'])
        reward_sm = shared_memory.SharedMemory(create=True, size=sizes['reward_sb_size'])

        actions = np.ndarray((self._batch_size, *self._action_shape), dtype=np.float32, buffer=action_sm.buf)
        dones = np.ndarray((self._batch_size,), dtype=np.float32,buffer=done_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, buffer=reward_sm.buf)
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, dones=dones,
                      next_images=next_images, next_proprioceptions=next_propris)
        
        sm_names = {}
        if not self._ignore_image:
            sm_names['img_sm'] = img_sm.name
            sm_names['next_img_sm'] = next_img_sm.name
        if not self._ignore_propri:
            sm_names['proprioception_sm'] = proprioception_sm.name
            sm_names['next_proprioception_sm'] = next_proprioception_sm.name
        sm_names['action_sm'] = action_sm.name
        sm_names['done_sm'] = done_sm.name
        sm_names['reward_sm'] = reward_sm.name
        
        sms = (img_sm, next_img_sm,  proprioception_sm, 
               next_proprioception_sm, action_sm, done_sm, reward_sm)

        return sb, sms, sm_names

    def _get_sm_sb_sp(self, sm_names):
        total_size = 0
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(
                name=sm_names['img_sm'])
            next_img_sm = shared_memory.SharedMemory(
                name=sm_names['next_img_sm'])
            
            images = np.ndarray((self._batch_size, *self._stacked_image_shape), 
                                dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray((self._batch_size, *self._stacked_image_shape), 
                                    dtype=np.uint8, buffer=next_img_sm.buf)
            total_size += images.nbytes + next_images.nbytes

        propris = None
        next_propris = None    
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(name=sm_names['proprioception_sm'])
            next_proprioception_sm = shared_memory.SharedMemory(name=sm_names['next_proprioception_sm'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)
            
            total_size += propris.nbytes + next_propris.nbytes

        action_sm = shared_memory.SharedMemory(name=sm_names['action_sm'])
        done_sm = shared_memory.SharedMemory(name=sm_names['done_sm'])
        reward_sm = shared_memory.SharedMemory(name=sm_names['reward_sm'])
        
        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_sm.buf)
        dones = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=done_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_sm.buf)
        
        total_size += actions.nbytes + dones.nbytes + rewards.nbytes
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, dones=dones,
                      next_images=next_images, next_proprioceptions=next_propris)
              
        sms = (img_sm, next_img_sm,  proprioception_sm, 
                next_proprioception_sm, action_sm, done_sm, reward_sm)

        return sb, sms, total_size
    
    def _init_rb_sp(self):
        total_size = 0

        if self._load_path:
            total_size = self._load_sp()
        else:
            self._idx = 0
            self._image_idx = 0
            self._full = False
            self._count = 0 

            if not self._ignore_image:
                rb_images_size = self._height * self._width * \
                    self._image_capacity * np.dtype(np.uint8).itemsize
                self._rb_images_sm = shared_memory.SharedMemory(create=True, size=rb_images_size)
                
                if self._image_type == 'hwc':    
                    self._rb_images = np.ndarray(
                        (self._height, self._width, self._image_capacity), 
                        dtype=np.uint8, buffer=self._rb_images_sm.buf)
                else:
                    self._rb_images = np.ndarray(
                        (self._image_capacity, self._height, self._width), 
                        dtype=np.uint8, buffer=self._rb_images_sm.buf)
                    
                self._rb_image_references = np.empty((self._capacity, 2), dtype=np.int32)
                self._rb_next_image_references = np.empty((self._capacity, 2), dtype=np.int32)
                self._last_image_ref = np.array([-1, -1], dtype=np.int32)

                total_size += self._rb_images.nbytes + self._rb_image_references.nbytes + \
                    self._rb_next_image_references.nbytes + self._last_image_ref.nbytes

            if not self._ignore_propri:
                self._rb_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), dtype=np.float32)
                self._rb_next_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), dtype=np.float32)
                
                total_size += self._rb_propris.nbytes + self._rb_next_propris.nbytes

            self._rb_actions = np.empty((self._capacity, *self._action_shape), dtype=np.float32)
            self._rb_rewards = np.empty((self._capacity), dtype=np.float32)
            self._rb_dones = np.empty((self._capacity), dtype=np.float32)

            total_size += self._rb_actions.nbytes + self._rb_rewards.nbytes + self._rb_dones.nbytes

        return total_size

    def _produce_samples_sp(self): 
        workers = None
        worker_queues = None

        if not self._ignore_image:
            workers = []
            worker_queues = []
            self._worker_conf_queue = Queue() 
            self._image_capacity = (self._capacity * self._channels) + \
            (self._max_num_episodes * self._image_history * self._channels) + \
            (2 * self._max_episode_lenght * self._channels)

            self._image_threshold = self._image_capacity - (2 * self._max_episode_lenght * self._channels)
        
            for _ in range(self._num_workers):
                q = Queue()
                p = Process(target=self._sample_images_sp_worker, args=(q,))
                worker_queues.append(q)
                workers.append(p)
            
            for p in workers: 
                p.start()

            self._workers = workers
            self._worker_queues = worker_queues
            
        self._rcv_from_ip_queue = self._send_to_sampling_process_queue
        self._send_to_ip_queue = self._rcv_from_sampling_process_queue

        rb_size = self._init_rb_sp()

        self._recv_obs_thread = threading.Thread(target=self._recv_obs_sp)
        self._recv_obs_thread.start()

        sb_0_sm_names = self._rcv_from_ip_queue.get()
        sb_1_sm_names = self._rcv_from_ip_queue.get()

        if not self._ignore_image:
            for worker_queue in self._worker_queues:
                sb_0_sm_images_names = (sb_0_sm_names['img_sm'], sb_0_sm_names['next_img_sm'])
                sb_1_sm_images_names = (sb_1_sm_names['img_sm'], sb_1_sm_names['next_img_sm'])
                worker_queue.put((self._rb_images_sm.name, sb_0_sm_images_names, sb_1_sm_images_names))

        self._sb_0, self._sb_0_sm, sb_0_size = self._get_sm_sb_sp(sb_0_sm_names)
        self._sb_1, self._sb_1_sm, sb_1_size = self._get_sm_sb_sp(sb_1_sm_names)

        total_size = rb_size + sb_0_size + sb_1_size
        print(f'Total size of buffers (in GB): {total_size / 1e9}')

        self._send_to_ip_queue.put(READY)

        while not self._start_batch:
            # Checking if the replay buffer process 
            # needs to be closed while waiting
            if not self._rcv_from_ip_queue.empty():
                code = self._rcv_from_ip_queue.get()
                if code == CLOSE:
                    self._close_sp()
                    return
                elif code == SAVE:
                    self._rcv_from_ip_queue.get()
            time.sleep(0.1)

        with self._lock:
            self._sample_sp(self._sb_0, SB_0)
            self._sample_sp(self._sb_1, SB_1)

        self._send_to_ip_queue.put(SB_0)
        self._send_to_ip_queue.put(SB_1)

        while True:
            code = self._rcv_from_ip_queue.get()
            if code == SB_0:
                with self._lock:
                    self._sample_sp(self._sb_0, SB_0)
                self._send_to_ip_queue.put(SB_0)
            elif code == SB_1:
                with self._lock:
                    self._sample_sp(self._sb_1, SB_1)
                self._send_to_ip_queue.put(SB_1)
            elif code == SAVE:
                save_path = self._rcv_from_ip_queue.get()
                self._save_sp(save_path)
            elif code == CLOSE:
                break

        self._close_sp()

    def _insert_image_to_buffer_sp(self, image, first_step_of_episode=False):
        index = self._image_idx
        if first_step_of_episode and index > self._image_threshold:
            index = 0

        if first_step_of_episode:
            if self._image_type == 'hwc':
                for i in range(self._image_history): 
                    self._rb_images[:, :, index+(i*self._channels):index+((i+1)*self._channels)] = image
            else:
                for i in range(self._image_history):
                    self._rb_images[index+(i*self._channels):index+((i+1)*self._channels), :, :] = image
            self._image_idx = index + (self._image_history * self._channels)
            return np.array([index, self._image_idx])

        if self._image_type == 'hwc':
            self._rb_images[:, :, index:index+self._channels] = image
        else:
            self._rb_images[index:index+self._channels, :, :] = image
        self._image_idx = index + self._channels
                
        return np.array([self._image_idx-(self._image_history*self._channels), self._image_idx])
        
    def _add_sp(self, 
            image, 
            propri, 
            action, 
            reward, 
            next_image, 
            next_propri, 
            done,
            first_step_of_episode):
        
        if not self._ignore_image:
            if first_step_of_episode:
                self._last_image_ref = self._insert_image_to_buffer_sp(image, first_step_of_episode)
            
            self._rb_image_references[self._idx] = self._last_image_ref
            self._last_image_ref = self._insert_image_to_buffer_sp(next_image)
            self._rb_next_image_references[self._idx] = self._last_image_ref

        if not self._ignore_propri:
            self._rb_propris[self._idx] = propri
            self._rb_next_propris[self._idx] = next_propri

        self._rb_actions[self._idx] = action
        self._rb_rewards[self._idx] = reward
        self._rb_dones[self._idx] = done

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0
        self._count = self._capacity if self._full else self._idx    

    def _sample_images_sp_worker(self, rcv_queue):
        initial_data = rcv_queue.get()
        rb_images_sm_name, sb_0_names, sb_1_names = initial_data

        sb_0_images_sm_name, sb_0_next_images_sm_name = sb_0_names
        sb_1_images_sm_name, sb_1_next_images_sm_name = sb_1_names

        rb_images_sm = shared_memory.SharedMemory(name=rb_images_sm_name)
        sb_0_images_sm = shared_memory.SharedMemory(name=sb_0_images_sm_name)
        sb_0_next_images_sm = shared_memory.SharedMemory(name=sb_0_next_images_sm_name)
        sb_1_images_sm = shared_memory.SharedMemory(name=sb_1_images_sm_name)
        sb_1_next_images_sm = shared_memory.SharedMemory(name=sb_1_next_images_sm_name) 

        if self._image_type == 'hwc':    
            rb_images = np.ndarray(
                (self._height, self._width, self._image_capacity), 
                dtype=np.uint8, buffer=rb_images_sm.buf)
        else:
            rb_images = np.ndarray(
                (self._image_capacity, self._height, self._width), 
                dtype=np.uint8, buffer=rb_images_sm.buf)
        
        sb_0_images = np.ndarray((self._batch_size, *self._stacked_image_shape),
                                 dtype=np.uint8, buffer=sb_0_images_sm.buf)
        sb_0_next_images = np.ndarray((self._batch_size, *self._stacked_image_shape), 
                                      dtype=np.uint8, buffer=sb_0_next_images_sm.buf)
        sb_1_images = np.ndarray((self._batch_size, *self._stacked_image_shape),
                                 dtype=np.uint8, buffer=sb_1_images_sm.buf)
        sb_1_next_images = np.ndarray((self._batch_size, *self._stacked_image_shape), 
                                      dtype=np.uint8, buffer=sb_1_next_images_sm.buf)
        
        while True:
            ins = rcv_queue.get() 
            if ins == CLOSE:
                rb_images_sm.close()
                sb_0_images_sm.close()
                sb_0_next_images_sm.close()
                sb_1_images_sm.close()
                sb_1_next_images_sm.close()
                print('..closed')
                return
            
            sb_code, rb_images_indices,  rb_next_images_indices, sb_indices_range = ins
            sb_start_index, sb_end_index = sb_indices_range

            j = 0
            for i in range(sb_start_index, sb_end_index):
                rb_images_start_index, rb_images_end_index = rb_images_indices[j][:]
                rb_next_images_start_index, rb_next_images_end_index = rb_next_images_indices[j][:]
                j += 1
                if sb_code == SB_0:
                    if self._image_type == 'hwc':
                        sb_0_images[i] = rb_images[:, :, rb_images_start_index:rb_images_end_index]
                        sb_0_next_images[i] = rb_images[:, :, rb_next_images_start_index:rb_next_images_end_index]
                    else:
                        sb_0_images[i] = rb_images[rb_images_start_index:rb_images_end_index, :, :]
                        sb_0_next_images[i] = rb_images[rb_next_images_start_index:rb_next_images_end_index, :, :]
                else:
                    if self._image_type  == 'hwc': 
                        sb_1_images[i] = rb_images[:, :, rb_images_start_index:rb_images_end_index]
                        sb_1_next_images[i] = rb_images[:, :, rb_next_images_start_index:rb_next_images_end_index]
                    else:
                        sb_1_images[i] = rb_images[rb_images_start_index:rb_images_end_index, :, :]
                        sb_1_next_images[i] = rb_images[rb_next_images_start_index:rb_next_images_end_index, :, :]

            self._worker_conf_queue.put(1)
            
    def _sample_sp(self, sb, sb_name):
        idxs = np.random.randint(0, self._count,
                                 size=min(self._count, self._batch_size))

        if not self._ignore_image:            
            image_indices = self._rb_image_references[idxs]
            next_image_indices = self._rb_next_image_references[idxs]

            for i in range(self._num_workers):
                start, end = i * self._worker_batch_size, (i + 1) * self._worker_batch_size
                worker_ins = (sb_name, 
                              image_indices[start:end, :],
                              next_image_indices[start:end, :],
                              (start, end)) 
                self._worker_queues[i].put(worker_ins)

        if not self._ignore_propri:
            np.copyto(sb.proprioceptions, self._rb_propris[idxs])
            np.copyto(sb.next_proprioceptions, self._rb_next_propris[idxs])

        np.copyto(sb.actions, self._rb_actions[idxs])
        np.copyto(sb.rewards, self._rb_rewards[idxs])
        np.copyto(sb.dones, self._rb_dones[idxs])

        if not self._ignore_image:
            for i in range(self._num_workers):
                self._worker_conf_queue.get()

    def _close_sp(self):
        self._obs_queue.put(CLOSE)

        if not self._ignore_image:
            for worker_queue in self._worker_queues:
                worker_queue.put(CLOSE)

            for worker_process in self._workers:
                worker_process.join()

            self._rb_images_sm.close()
            self._rb_images_sm.unlink()

        print('Closng replay buffer shared memory..')
        with self._lock:
            for sm in self._sb_0_sm:
                if sm is not None:
                    try:
                        sm.close()
                    except:
                        pass
            for sm in self._sb_1_sm:
                if sm is not None:
                    try:
                        sm.close()
                    except:
                        pass 


def get_buffer_size(
        image_shape,
        proprioception_shape,
        action_shape,
        capacity,
        batch_size,
        image_history,
        max_num_episodes,
        max_episode_lenght,
        image_type='hwc'):
    
    total_image_size = 0
    total_proprioception_size = 0
    total_action_size = 0
    total_reward_size = 0
    total_done_size = 0


    if image_shape:
        if image_type == 'hwc':
            height, width, channels = image_shape
            stacked_image_shape = (height, width, channels * image_history)
        else:
            channels, height, width = image_shape
            stacked_image_shape = (channels * image_history, height, width)

        sb_image_size = np.dtype(np.uint8).itemsize * batch_size * np.prod(stacked_image_shape)
        sb_image_size *= 4

        image_capacity = (capacity * channels) + (max_num_episodes * image_history * channels) + \
            (2 * max_episode_lenght * channels)
        
        rb_images_size = height * width * image_capacity * np.dtype(np.uint8).itemsize
        rb_image_references_size = np.dtype(np.int32).itemsize * capacity * 2
        rb_next_image_references_size = rb_image_references_size
        last_image_ref_size = np.dtype(np.int32).itemsize * 2

        total_image_size = sb_image_size + rb_images_size + rb_image_references_size + \
            rb_next_image_references_size + last_image_ref_size

    if proprioception_shape:
        sb_proprioception_size = np.dtype(np.float32).itemsize * batch_size * np.prod(proprioception_shape)
        sb_proprioception_size *= 4
        rb_proprioception_size = np.dtype(np.float32).itemsize * capacity * np.prod(proprioception_shape)
        rb_proprioception_size *= 2
        total_proprioception_size = sb_proprioception_size + rb_proprioception_size

    sb_action_size = np.dtype(np.float32).itemsize * batch_size * np.prod(action_shape)
    sb_action_size *= 2
    rb_action_size = np.dtype(np.float32).itemsize * capacity * np.prod(action_shape)
    total_action_size = sb_action_size + rb_action_size

    sb_reward_size = np.dtype(np.float32).itemsize * batch_size
    sb_reward_size *= 2
    rb_reward_size = np.dtype(np.float32).itemsize * capacity
    total_reward_size = sb_reward_size + rb_reward_size

    sb_done_size = np.dtype(np.float32).itemsize * batch_size
    sb_done_size *= 2
    rb_done_size = np.dtype(np.float32).itemsize * capacity
    total_done_size = sb_done_size + rb_done_size

    total = total_image_size + total_proprioception_size + total_action_size + total_reward_size + total_done_size

    return total

