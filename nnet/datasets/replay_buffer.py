# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# NeuralNets
from nnet import datasets
from nnet import utils

# Other
import os
import random
import collections
import glob

class ReplayBuffer(datasets.Dataset):

    def __init__(
            self,
            batch_size, 
            root, 
            buffer_capacity,
            epoch_length, 
            sample_length,
            shuffle=True,
            save_trajectories=True, 
            collate_fn=utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[]), 
            buffer_name="ReplayBuffer"
        ):
        super(ReplayBuffer, self).__init__(num_workers=0, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=root)

        # Params
        self.buffer_name = buffer_name
        self.buffer_capacity = buffer_capacity
        self.epoch_length = epoch_length
        self.sample_length = sample_length
        self.ram_buffer = collections.OrderedDict()
        self._sample_keys = None
        self.streams = collections.OrderedDict()
        self.traj_index = torch.tensor(0)
        self.num_steps = torch.tensor(0)
        self.last_save_traj_index = 0
        self.buffer_dir = os.path.join(root, self.buffer_name)
        self.save_trajectories = save_trajectories
        self.retrieval = None
        self.source_dirs = None
        self.source_files = []

        # Create Buffer Dir
        if self.save_trajectories and not os.path.isdir(self.buffer_dir):
            os.makedirs(self.buffer_dir, exist_ok=True)

    def get_infos(self):
        return { 
            "traj_index": self.traj_index,
            "num_steps": self.num_steps
        }

    def state_dict(self):
        return { 
            "traj_index": self.traj_index,
            "num_steps": self.num_steps,
            "buffer_keys": list(self.ram_buffer.keys()),
            **({"source_dirs": list(dict.fromkeys(self.source_dirs + [os.path.abspath(self.buffer_dir)]))}
               if self.source_dirs is not None else {}),
            **({"source_files": list(dict.fromkeys(self.source_files + [os.path.abspath(
                os.path.join(self.buffer_dir, "{}.torch".format(self.traj_index)))]))}
               if self.source_dirs is not None else {}),
            **({"retrieval": self.retrieval.state_dict()} if self.retrieval is not None else {})
        }
    
    def load_state_dict(self, state_dict):
        self.source_dirs = state_dict.get("source_dirs")
        self.source_files = state_dict.get("source_files")
        self.traj_index.fill_(state_dict.pop("traj_index"))
        self.num_steps.fill_(state_dict.pop("num_steps"))
        self.load(state_dict.pop("buffer_keys"))
        if self.source_dirs is not None:
            self.streams.clear()
        if self.retrieval is not None:
            self.streams.clear()
            self.retrieval.restore(state_dict.get("retrieval"))

    def enable_retrieval(self, num_envs):
        from .retrieval_replay import RetrievalReplay
        self.retrieval = RetrievalReplay(self, num_envs)
        self.collate_fn = utils.CollateFn(
            inputs_params=[{"axis": axis} for axis in range(7)], targets_params=[])
        if self.ram_buffer:
            self.retrieval.restore()

    def retrieval_view(self):
        return self.retrieval

    def save(self):

        # Save Trajs
        if self.save_trajectories:

            # Select Trajs
            save_trajs = {traj_id:traj for traj_id, traj in self.ram_buffer.items()
                          if traj_id >= self.last_save_traj_index}

            # Save Trajs
            path = os.path.abspath(os.path.join(self.buffer_dir, "{}.torch".format(self.traj_index)))
            # A final checkpoint can immediately follow a periodic save. Do not
            # replace that trajectory file with an empty incremental save.
            if self.last_save_traj_index != self.traj_index.item() or not os.path.isfile(path):
                torch.save(save_trajs, path)
            if path not in self.source_files:
                self.source_files.append(path)

            # Update 
            self.last_save_traj_index = self.traj_index.item()

    def load(self, buffer_keys):
        self._sample_keys = None
        
        # All Saves
        # A shared checkpoint names its exact sources. Existing files in a new
        # branch's output directory must not override that shared history.
        directories = list(dict.fromkeys(self.source_dirs if self.source_dirs is not None else [self.buffer_dir]))
        paths = (self.source_files if self.source_dirs is not None and self.source_files is not None else
                 [os.path.abspath(path) for directory in directories for path in glob.glob(os.path.join(directory, "*.torch"))])
        required = set(buffer_keys)
        for path_trajs in paths:

            # Load Save
            load_trajs = torch.load(path_trajs)

            # Add required trajs
            for key, value in load_trajs.items():
                if key in required:
                    self.ram_buffer[key] = value

        # Assert all keys loaded
        assert sorted(buffer_keys) == sorted(list(self.ram_buffer.keys())), "some buffer traj keys are missing, buffer save may be corrupted: {} buffer keys, {} loaded keys".format(len(buffer_keys), len(list(self.ram_buffer.keys())))

        # Update 
        self.last_save_traj_index = self.traj_index.item()
        self.source_files = list(paths)

    def enforce_capacity(self):

        # Pop episodes
        while self.num_steps > self.buffer_capacity:

            # Pop oldest Episode
            oldest_episode_id = (self.traj_index - self.num_steps).item()
            self.ram_buffer.pop(oldest_episode_id)
            self._sample_keys = None
            if self.retrieval is not None:
                self.retrieval.remove_window(oldest_episode_id)

            # Update Number of steps
            self.num_steps -= 1

    def append_step(self, sample, sample_id):

        # None sample
        if sample is None:
            return

        # Init Stream
        if sample_id not in self.streams:
            self.streams[sample_id] = []

        # Select stream
        stream = self.streams[sample_id]

        # Update Stream
        stream.append([s.clone() for s in sample]) # Clone

        # Unfinished Trajectory
        if len(stream) < self.sample_length:
            return self.get_infos()
        assert len(stream) == self.sample_length

        # Order Traj (elt, time) without memory copy
        traj = [[stream[t][elt] for t in range(self.sample_length)] for elt in range(len(stream[0]))]

        # Slice Stream first element
        self.streams[sample_id].pop(0)

        # Add to ram buffer (using tensor instead of int as key will replace instead of adding)
        self.ram_buffer[self.traj_index.item()] = traj
        self._sample_keys = None
        if self.retrieval is not None:
            self.retrieval.add_window(self.traj_index.item(), sample_id, traj)

        # Update Index
        self.traj_index += 1
        self.num_steps += 1

        # enforse num_steps <= buffer_capacity
        self.enforce_capacity()

        return self.get_infos()

    def __len__(self):

        return self.epoch_length * self.batch_size

    def __getitem__(self, n):

        # Sample
        sample = self.sample()

        return sample
    
    def sample(self):

        
        # Select Episode from ram
        # Keep the same insertion order and random.choice call. Rebuild only
        # after append, eviction or load changes the eligible trajectories.
        if self._sample_keys is None:
            self._sample_keys = list(self.ram_buffer.keys())
        traj_id = random.choice(self._sample_keys)
        traj = self.ram_buffer[traj_id]

        # Stack elts
        traj = [torch.stack(elt, axis=0) for elt in traj]

        if self.retrieval is not None:
            traj.append(torch.tensor(self.retrieval.windows[traj_id], dtype=torch.int64))

        return traj
