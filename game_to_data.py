import peppi_py
import pyarrow as pa
import numpy as np
import argparse
import os
import p_consts
from scipy.spatial.distance import cdist


"""
Pass a peppi_py.game object to the flatten_game_to_numpy function and it will return a dict which maps 
states/actions to all their values for every frame.  Bottom of the file there are two lists which specify
exactly which features are in the training and label data.  Finally, the np arrays S and A are the respective
state and action arrays.  S is shifted relative to actions by one frame so that the state of one frame can be
used to predict the next action.

Code is all still very rough --> not generalized for large project and works as simple local script, i still
need to make it functional :(
"""

LEAF_TYPES = (pa.Array, pa.ChunkedArray, pa.Scalar)

def _flatten_recursive(obj, prefix, data_dict):
    """Internal recursive helper function."""
    
    if isinstance(obj, LEAF_TYPES):
        data_dict[prefix] = np.array(obj)
        return

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if item is not None: 
                new_prefix = f"{prefix}_{i}"
                _flatten_recursive(item, new_prefix, data_dict)
    
    elif hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
        for attr_name in dir(obj):
            if not attr_name.startswith('_'): 
                attr_val = getattr(obj, attr_name)
                
                if attr_val is None or callable(attr_val) or attr_name == 'type':
                    continue
                
                new_prefix = f"{prefix}_{attr_name}" if prefix else attr_name
                _flatten_recursive(attr_val, new_prefix, data_dict)

def flatten_game_to_numpy(game):
    """
    Takes a loaded peppi_py game object and returns a single, flat
    dictionary where keys are the feature paths and values are
    full-length NumPy arrays.
    """
    flat_data = {}
    
    _flatten_recursive(game.frames, '', flat_data)
    
    # We rename 'ports_0' -> 'p0', 'ports_1' -> 'p1' for clarity
    final_data = {}
    for key, val in flat_data.items():
        new_key = key.replace('ports_0', 'p0').replace('ports_1', 'p1')
        new_key = new_key.replace('ports_2', 'p2').replace('ports_3', 'p3')
        final_data[new_key] = val
        
    return final_data

def discretize(joy_x : str, joy_y: str) -> np.ndarray:
    """
    Pass in the keys for dict corresponding to joystick x and joystick y.
    
    Used for both the left stick and the c stick
    """
    raw_inputs = np.stack([np_data[joy_x], np_data[joy_y]], axis = 1)
    distances = cdist(raw_inputs, p_consts.STICK_XY_CLUSTER_CENTERS_V1, metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    snapped_coordinates = p_consts.STICK_XY_CLUSTER_CENTERS_V1[nearest_indices]
    return snapped_coordinates






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a Slippi .slp game file into numpy arrays for states/actions.")
    parser.add_argument("--file", help="Path to the .slp game file (default: Game_20251015T011631.slp)", required=True)
    parser.add_argument("--debug", help="Enable debug output", default=True, action="store_true")
    args = parser.parse_args()
    
    # Load your game
    game = peppi_py.read_slippi(args.file)
    np_data = flatten_game_to_numpy(game)

    if args.debug:
        print(f"Flattened {len(np_data.keys())} arrays.")
        print(f"P1's X-Position array (shape): {np_data['p0_leader_post_position_x'].shape}")
        print(f"P2's Buttons array (shape): {np_data['p1_leader_pre_buttons'].shape}")
    
    disc_left_stick = discretize('p0_leader_pre_joystick_x', 'p0_leader_pre_joystick_y')
    disc_c_stick = discretize('p0_leader_pre_cstick_x', 'p0_leader_pre_cstick_y')

    post_data_full = np.stack(
        [np_data[key] for key in p_consts.POST_STATE_FEATURES], axis=1
    )

    #just need to fix the buttons, but the sticks should hopefully be discretized :)
    pre_data_full = np.stack([
        disc_left_stick,
        disc_c_stick,
        np_data['p0_leader_pre_triggers_physical_l'],
        np_data['p0_leader_pre_triggers_physical_r'],
        np_data['p0_leader_pre_buttons_physical']
    ],
        axis=1
        
    )

    if args.debug:
        print(f"\nFull Post-State array shape: {post_data_full.shape}")
        print(f"Full Pre-Action array shape: {pre_data_full.shape}")

    # S = All post-frames *except the last one*
    S = post_data_full[:-1]

    # A = All pre-frames *except the first one*
    A = pre_data_full[1:]

    if args.debug:
        print(f"\nFinal State (S) shape: {S.shape}")
        print(f"Final Action (A) shape: {A.shape}")
        print(S)

    # Now S[i] is perfectly aligned with A[i]
    # (i.e., post[i] is aligned with pre[i+1])
    # You can now save S and A to your HDF5 file.

    # S[9] is the 10th frame
    # A[10] is the 11th action taken from inputs on the 10th frame