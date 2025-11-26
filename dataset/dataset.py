import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from .discretizer import ActionDiscretizer
import peppi_py
from game_to_data import flatten_game_to_numpy
import p_consts
from scipy.spatial.distance import cdist


MARTH_CHARACTER_ID = 9

class GameDataset(Dataset):
    def __init__(self, game_dir, seq_len=16):
        # seq_len: Number of frames to use as input sequence
        self.game_dir = Path(game_dir)
        self.seq_len = seq_len
        self.discretizer = ActionDiscretizer()
        
        self.all_states = []
        self.all_actions = []
        
        self._load_all_games()
        
    def _load_all_games(self):
        """Load all .slp files and concatenate them."""
        
        slp_files = sorted(self.game_dir.glob("*.slp"))
        if not slp_files:
            raise ValueError(f"No .slp files found in {self.game_dir}")
        
        print(f"Loading {len(slp_files)} .slp files...")
        
        for slp_file in slp_files:
            try:
                game = peppi_py.read_slippi(str(slp_file))
                np_data = flatten_game_to_numpy(game)
                
                # Discretize joystick and c-stick using the cluster centers
                # (following the same approach as game_to_data.py)
                
                # Discretize left stick
                raw_left = np.stack([np_data['p0_leader_pre_joystick_x'], 
                                    np_data['p0_leader_pre_joystick_y']], axis=1)
                distances_left = cdist(raw_left, p_consts.STICK_XY_CLUSTER_CENTERS_V1, metric='euclidean')
                nearest_left = np.argmin(distances_left, axis=1)
                disc_left_stick = p_consts.STICK_XY_CLUSTER_CENTERS_V1[nearest_left]
                
                # Discretize c-stick
                raw_c = np.stack([np_data['p0_leader_pre_cstick_x'], 
                                 np_data['p0_leader_pre_cstick_y']], axis=1)
                distances_c = cdist(raw_c, p_consts.STICK_XY_CLUSTER_CENTERS_V1, metric='euclidean')
                nearest_c = np.argmin(distances_c, axis=1)
                disc_c_stick = p_consts.STICK_XY_CLUSTER_CENTERS_V1[nearest_c]
                
                # Extract post-state features (48 features: 24 per player)
                post_data = np.stack([np_data[key] for key in p_consts.POST_STATE_FEATURES], axis=1)
                
                # Build pre-action features with discretized sticks
                pre_data = np.stack([
                    disc_left_stick[:, 0],   # joystick_x (discretized)
                    disc_left_stick[:, 1],   # joystick_y (discretized)
                    disc_c_stick[:, 0],      # cstick_x (discretized)
                    disc_c_stick[:, 1],      # cstick_y (discretized)
                    np_data['p0_leader_pre_triggers_physical_l'],
                    np_data['p0_leader_pre_triggers_physical_r'],
                    np_data['p0_leader_pre_buttons_physical']
                ], axis=1)
                
                states = post_data[:-1]  # [num_frames, 48]
                actions = pre_data[1:]   # [num_frames, 7]
                
                self.all_states.append(states)
                self.all_actions.append(actions)
                
            except Exception as e:
                print(f"Error loading {slp_file.name}: {e}")
                continue
        
        self.all_states = np.concatenate(self.all_states, axis=0)
        self.all_actions = np.concatenate(self.all_actions, axis=0)
        
        print("Discretizing actions...")
        self.all_actions_disc = self.discretizer.batch_discretize(self.all_actions)
        
        self.all_states = torch.FloatTensor(self.all_states)
        
        print(f"Total frames: {len(self.all_states):,}")
        print(f"Total sequences: {len(self.all_states) - self.seq_len:,}")
        
    def __len__(self):
        return len(self.all_states) - self.seq_len
    
    def __getitem__(self, idx):
        """
        Returns:
            state_seq: [seq_len, 48] - sequence of states
            action_targets: dict with discretized action components
        """
        state_seq = self.all_states[idx:idx + self.seq_len]  # [seq_len, 48]
        
        # Get target action (action taken after last state in sequence)
        target_idx = idx + self.seq_len
        action_targets = {
            'joystick': torch.LongTensor([self.all_actions_disc['joystick'][target_idx]]),
            'cstick': torch.LongTensor([self.all_actions_disc['cstick'][target_idx]]),
            'trigger_l': torch.LongTensor([self.all_actions_disc['trigger_l'][target_idx]]),
            'trigger_r': torch.LongTensor([self.all_actions_disc['trigger_r'][target_idx]]),
            'buttons': torch.FloatTensor(self.all_actions_disc['buttons'][target_idx])
        }
        
        return state_seq, action_targets


def create_dataloader(game_dir, seq_len=16, batch_size=32, train_split=0.8, num_workers=0):
    dataset = GameDataset(game_dir=game_dir, seq_len=seq_len)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print(f"\nDataset split:")
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader
