import numpy as np
import collections
import threading
import time
import os
import pickle
from multiprocessing import shared_memory, Process, Queue, Lock


Batch = collections.namedtuple(
    'Batch', ['images', 'proprioceptions', 'actions', 'rewards',
              'masks', 'next_images', 'next_proprioceptions'])


class RadReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, image_shape, proprioception_shape, action_shape,
                 capacity, batch_size, init_buffers=True, load_path=''):

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

        if image_shape[-1] != 0:
            self._ignore_image = False

        if proprioception_shape[-1] != 0:
            self._ignore_propri = False

        if load_path:
            self._load_path = load_path
        else:
            self._load_path = ''

        if init_buffers:
            self._init_buffers()

        self._target_sizes = {}

    def _init_buffers(self):
        if self._load_path:
            self._load()
        else:
            if not self._ignore_image:
                self._images = np.empty(
                    (self._capacity, *self._image_shape), dtype=np.uint8)
                self._next_images = np.empty(
                    (self._capacity, *self._image_shape), dtype=np.uint8)

            if not self._ignore_propri:
                self._propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                self._next_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)

            self._actions = np.empty(
                (self._capacity, *self._action_shape), dtype=np.float32)
            
            self._rewards = np.empty((self._capacity), dtype=np.float32)
            self._masks = np.empty((self._capacity), dtype=np.float32)

    def add(self, image, propri, action, reward, next_image, next_propri, mask):
        if not self._ignore_image:
            self._images[self._idx] = image
            self._next_images[self._idx] = next_image
        if not self._ignore_propri:
            self._propris[self._idx] = propri
            self._next_propris[self._idx] = next_propri
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward

        if self._idx in self._target_sizes:
            self._target_sizes.pop(self._idx)

        if type(mask) is tuple:
            self._masks[self._idx] = mask[0]
            self._target_sizes[self._idx] = mask[1]
        else:
            self._masks[self._idx] = mask

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0
        self._count = self._capacity if self._full else self._idx
        self._steps += 1

    def sample(self):
        idxs = np.random.randint(0, self._count,
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
        masks = self._masks[idxs]

        return Batch(images=images, proprioceptions=propris,
                     actions=actions, rewards=rewards, masks=masks,
                     next_images=next_images, next_proprioceptions=next_propris)

    def update_masks(self, target_size):
        for key, value in self._target_sizes.items():
            if value < target_size:
                self._masks[key] = 1.0
    

    def save(self, save_path):
        tic = time.time()
        print(f'Saving the replay buffer in {save_path}..')
        with self._lock:
            data = {
                'count': self._count,
                'idx': self._idx,
                'full': self._full,
                'steps': self._steps,
                'target_sizes': self._target_sizes
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
            np.save(os.path.join(save_path, "masks.npy"), self._masks)

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
        self._target_sizes = data['target_sizes'] 

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
        self._masks = np.load(os.path.join(self._load_path, "masks.npy"))

        print("Loaded the buffer from: {}".format(self._load_path), end=' ')
        print("Took: {:.3f}s".format(time.time() - tic))


class AsyncSMRadReplayBuffer(RadReplayBuffer):
    def __init__(self, image_shape, proprioception_shape, action_shape, 
                 capacity, batch_size, obs_queue, init_steps, load_path=''):
        super().__init__(
            image_shape, proprioception_shape, action_shape, capacity,
            batch_size, False, load_path)
        
        sizes = self._get_batch_data_sizes(batch_size)

        self._init_steps = init_steps
        self._obs_queue = obs_queue

        self._producer_queue = Queue()
        self._consumer_queue = Queue()

        self._start_batch = False

        self._batch0_code = 0
        self._batch1_code = 1
        self._save_code = 2
        self._close_code = 3

        self._producer_process = Process(target=self._produce_batches)
        self._producer_process.start()

        self._batch0, self._batch0mem, mem_names0 = self._get_sm_batch(sizes)
        self._batch1, self._batch1mem, mem_names1 = self._get_sm_batch(sizes)

        self._consumer_queue.put(mem_names0)
        self._consumer_queue.put(mem_names1)

        self._last_batch = -1


    def sample(self):
        if not self._start_batch:
            self._start_batch = True
            self._obs_queue.put('start')
        batch_code = self._producer_queue.get()

        if self._last_batch != -1:
            self._consumer_queue.put(self._last_batch)

        self._last_batch = batch_code

        if batch_code == self._batch0_code:
            batch = self._batch0
        else:
            batch = self._batch1

        return batch
    
    def _recv_obs(self):
        while True:
            observation = self._obs_queue.get()
            if isinstance(observation, str):
                if observation == 'close':
                    return
                elif observation == 'start':
                    self._start_batch = True
                    continue
                elif observation == 'update_masks':
                    target_size = self._obs_queue.get()
                    with self._lock:
                        self.update_masks(target_size)
                    continue

            with self._lock:
                self.add(*observation)

    def _get_batch_data_sizes(self, batch_size):
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
        
        mask_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes
        
        reward_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes

        return image_size, proprioception_size, action_size, \
                mask_size, reward_size

    def _get_sm_batch(self, sizes):        
        image_size, proprioception_size, action_size, \
            mask_size, reward_size = sizes
        
        images = None
        next_images = None
        img_mem = None
        next_img_mem = None
        if not self._ignore_image:
            img_mem = shared_memory.SharedMemory(create=True, size=image_size)
            next_img_mem = shared_memory.SharedMemory(create=True, 
                                                      size=image_size)
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_mem.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_mem.buf)

        propris = None
        next_propris = None
        proprioception_mem = None
        next_proprioception_mem = None
        if not self._ignore_propri:
            proprioception_mem = shared_memory.SharedMemory(
                create=True, size=proprioception_size)
            next_proprioception_mem = shared_memory.SharedMemory(
                create=True, size=proprioception_size)
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_mem.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_mem.buf)


        action_mem = shared_memory.SharedMemory(create=True, size=action_size)
        mask_mem = shared_memory.SharedMemory(create=True, size=mask_size)
        reward_mem = shared_memory.SharedMemory(create=True, size=reward_size)

        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_mem.buf)
        masks = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=mask_mem.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_mem.buf)
        
        batch = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, masks=masks,
                      next_images=next_images, next_proprioceptions=next_propris)
        
        mem_names = {}
        if not self._ignore_image:
            mem_names['img_mem'] = img_mem.name
            mem_names['next_img_mem'] = next_img_mem.name
        if not self._ignore_propri:
            mem_names['proprioception_mem'] = proprioception_mem.name
            mem_names['next_proprioception_mem'] = next_proprioception_mem.name
        mem_names['action_mem'] = action_mem.name
        mem_names['mask_mem'] = mask_mem.name
        mem_names['reward_mem'] = reward_mem.name
        
        mems = (img_mem, next_img_mem,  proprioception_mem, 
                next_proprioception_mem, action_mem, mask_mem, reward_mem)

        return batch, mems, mem_names
    
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
        np.copyto(batch_dest.masks, batch_src.masks)

    def _get_sm_batch_using_names(self, mem_names):
        images = None
        next_images = None
        img_mem = None
        next_img_mem = None
        if not self._ignore_image:
            img_mem = shared_memory.SharedMemory(
                name=mem_names['img_mem'])
            next_img_mem = shared_memory.SharedMemory(
                name=mem_names['next_img_mem'])
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_mem.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_mem.buf)

        propris = None
        next_propris = None    
        proprioception_mem = None
        next_proprioception_mem = None
        if not self._ignore_propri:
            proprioception_mem = shared_memory.SharedMemory(
                name=mem_names['proprioception_mem'])
            next_proprioception_mem = shared_memory.SharedMemory(
                name=mem_names['next_proprioception_mem'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_mem.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_mem.buf)

        action_mem = shared_memory.SharedMemory(
            name=mem_names['action_mem'])
        mask_mem = shared_memory.SharedMemory(
            name=mem_names['mask_mem'])
        reward_mem = shared_memory.SharedMemory(
            name=mem_names['reward_mem'])
        
        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_mem.buf)
        masks = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=mask_mem.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_mem.buf)
        
        batch = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, masks=masks,
                      next_images=next_images, next_proprioceptions=next_propris)
              
        mems = (img_mem, next_img_mem,  proprioception_mem, 
                next_proprioception_mem, action_mem, mask_mem, reward_mem)

        return batch, mems


    def _produce_batches(self): 
        self._init_buffers()
        self._recv_obs_thread = threading.Thread(target=self._recv_obs)
        self._recv_obs_thread.start()

        mem_names0 = self._consumer_queue.get()
        mem_names1 = self._consumer_queue.get()

        self._batch0, self._batch0mem = self._get_sm_batch_using_names(mem_names0)
        self._batch1, self._batch1mem = self._get_sm_batch_using_names(mem_names1)

        while not self._start_batch:
            # Checking if the replay buffer process 
            # needs to be close while waiting
            if not self._consumer_queue.empty():
                code = int(self._consumer_queue.get())
                if code == self._close_code:
                    self._close()
                    return
                elif code == self._save_code:
                    self._consumer_queue.get()
            time.sleep(0.1)

        with self._lock:
            batch0 = super().sample()
        self._copy_batch(batch0, self._batch0)

        with self._lock:
            batch1 = super().sample()
        self._copy_batch(batch1, self._batch1)

        self._producer_queue.put(self._batch0_code)
        self._producer_queue.put(self._batch1_code)

        while True:
            with self._lock:
                batch = super().sample()
            code = int(self._consumer_queue.get())
            if code == self._batch0_code:
                self._copy_batch(batch, self._batch0)
                self._producer_queue.put(self._batch0_code)
            elif code == self._batch1_code:
                self._copy_batch(batch, self._batch1)
                self._producer_queue.put(self._batch1_code)
            elif code == self._save_code:
                save_path = self._consumer_queue.get()
                super().save(save_path)
            elif code == self._close_code:
                break

        self._close()

    def _close(self):
        self._obs_queue.put('close')
        print('Closng replay buffer shared memory..')
        with self._lock:
            for mem in self._batch0mem:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass
            for mem in self._batch1mem:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass

        
    def save(self, save_path):
        self._consumer_queue.put(self._save_code)
        self._consumer_queue.put(save_path)
        time.sleep(1)
        self._lock.acquire()
        self._lock.release()

    def close(self):
        self._consumer_queue.put(self._close_code)
        self._producer_process.join()

        for mem in self._batch0mem:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass
        for mem in self._batch1mem:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass

