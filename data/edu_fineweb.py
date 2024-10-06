"""
Adapted from:
https://github.com/karpathy/build-nanogpt
"""

import torch
import numpy as np
import os
import subprocess

SHARDS_TRAIN = 99
SHARDS_VAL = 1
DATA_ROOT = './data/edu_fineweb10B'
TOKENIZE_SCRIPT = './data/edu_fineweb_tokenize.py'


class FineWebEduTokenizedDataset:

    def __init__(self, data_root=DATA_ROOT):
        if not os.path.isdir(data_root) or not os.path.isfile(data_root + '/edufineweb_val_000000.npy'):
            self.download_and_tokenize()

        files = os.listdir(data_root)
        self.train = sorted([f'{data_root}/{s}' for s in files if 'train' in s])
        self.val   = sorted([f'{data_root}/{s}' for s in files if 'val' in s])
        print(f"Found {len(self.train)} train shards and {len(self.val)} val shards")
        assert len(self.train) == SHARDS_TRAIN and len(self.val) == SHARDS_VAL, f"Expecting {SHARDS_TRAIN}, {SHARDS_VAL} shards bug got {len(self.train), len(self.val)}"


    def download_and_tokenize(self):
        print(f"Downloading and tokenizing fineweb-edu. This will take some time..")
        with subprocess.Popen(['python', TOKENIZE_SCRIPT]) as process:
            process.wait()


class DataLoaderLite:
    def __init__(self, shards, B, T):
        self.B = B
        self.T = T
        self.shards = shards
        self.current_shard = 0
        self.current_position = 0
        self.reset()

    def reset(self):
        self.set_state(0, 0)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T

        return x, y

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def get_state(self):
        return {'shard': self.current_shard, 'position': self.current_position}

    def set_state(self, shard, position):
        self.current_shard = shard
        self.current_position = position
        self.tokens = self.load_tokens(self.shards[self.current_shard])

    def __len__(self):
        approx_tokens = 10_000_000_000  # 10B
        return approx_tokens // (self.B * self.T)




if __name__ == '__main__':
    data = FineWebEduTokenizedDataset()
    train_loader = DataLoaderLite(data.train, B=8, T=1024)
    val_loader = DataLoaderLite(data.val, B=8, T=1024)
    context, targets = train_loader.next_batch()
    print('Sample batch:')
    print('context: ', context)
    print('targets: ', targets)
