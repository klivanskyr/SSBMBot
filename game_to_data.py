import peppi_py
import pyarrow as pa
import numpy as np
import argparse
import os
import p_consts
from scipy.spatial.distance import cdist
from typing import Tuple, List, Callable


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


#function may not be entirely consistent, havent checked it too deeply
#it is self consistent with attempts at breaking the task into smaller pieces and creating the same mapping
def create_final_action_events(bitmask_array: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Decodes the raw bitmask, aggregates redundant actions (JUMP/SHIELD), 
    and converts the sustained signal into discrete press events (using np.diff).
    
    Returns:
        A tuple containing the final (N, 10) array of press events and the column names.
    """
    
    final_names = [name for name, mask in p_consts.FINAL_ACTION_MAPPING]
    mask_values = np.array([mask for name, mask in p_consts.FINAL_ACTION_MAPPING])

    # 2. Decode and Aggregate (Step 1)
    
    # Vectorized Bitwise AND: (N, 1) & (1, 10) -> (N, 10)
    input_reshaped = bitmask_array.reshape(-1, 1)
    masks_np = mask_values.reshape(1, -1)
    
    decoded_mask_values = input_reshaped & masks_np
    
    # Convert to binary (1 for held, 0 for not held). This is the sustained signal.
    sustained_actions = (decoded_mask_values > 0).astype(np.int8)

    # 3. Create Events (Step 2: np.diff)
    
    # Pad the top with zeros to calculate the change on the first frame
    num_features = sustained_actions.shape[1]
    zeros = np.zeros((1, num_features), dtype=sustained_actions.dtype)
    padded_array = np.vstack([zeros, sustained_actions])
    
    # Calculate difference between frames
    change_array = np.diff(padded_array, axis=0)
    
    # Isolate only positive transitions (presses: 0 -> 1)
    press_events = (change_array > 0).astype(np.int8)
    
    return press_events, final_names

def _normalize_min_max(X: np.ndarray) -> np.ndarray:
    """
    Normalizes a feature array X to the range [0, 1], with optional inversion.
    """
    X_min = np.min(X)
    X_max = np.max(X)
    denominator = X_max - X_min
    
    if denominator == 0:
        return np.zeros_like(X, dtype=np.float32)
    
    # If inverse_scale is True, the numerator becomes (X_max - X)
    numerator = X - X_min
        
    X_norm = numerator / denominator
    
    return X_norm.astype(np.float32)

def _inverse_min_max(X: np.ndarray) -> np.ndarray:
    """
    Normalizes a feature array X to the range [0, 1], with optional inversion.
    """
    X_min = np.min(X)
    X_max = np.max(X)
    denominator = X_max - X_min
    
    if denominator == 0:
        return np.zeros_like(X, dtype=np.float32)
    
    
    numerator = X_max - X 
    X_norm = numerator / denominator
    
    return X_norm.astype(np.float32)


def _normalize_bipolar(X: np.ndarray) -> np.ndarray:
    """
    Normalizes a feature array X to the range [-1, 1].
    
    Args:
        X: The NumPy array column for a single feature (e.g., all X-position values).

    Returns:
        The normalized array (float32).
    """
    # Calculate min and max across the array
    X_min = np.min(X)
    X_max = np.max(X)
    
    denominator = X_max - X_min
    
    # Handle the edge case where the feature is constant
    if denominator == 0:
        # If constant, it's centered at zero, so all values become 0.
        return np.zeros_like(X, dtype=np.float32)
    
    # Apply the Bipolar formula: ((2 * (X - X_min)) / Range) - 1
    X_norm = (2 * (X - X_min) / denominator) - 1
    
    return X_norm.astype(np.float32)

def construct_post_data(np_data: dict) -> np.ndarray:
    normal = np.stack(
        [np_data[key] for key in p_consts.NORMALIZE_MIN_MAX], axis=1
    )
    inverse = np.stack(
        [np_data[key] for key in p_consts.INVERSE_MAPPING], axis=1
    )
    bip = np.stack(
        [np_data[key] for key in p_consts.BIPOLAR_MAPPING], axis=1
    )
    normal = _apply_func_columnwise(normal, _normalize_min_max)
    inverse = _apply_func_columnwise(inverse, _inverse_min_max)
    bip = _apply_func_columnwise(bip, _normalize_bipolar)
    S = np.concatenate([bip, inverse, normal], axis=1)
    return S

def _apply_func_columnwise(X_data: np.ndarray, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Applies a single-input function (like normalize_bipolar) to every column 
    of the input data array X_data.

    Args:
        X_data: The 2D NumPy array containing features to be normalized (e.g., S_bipolar_data).
        func: The normalization function to apply (e.g., normalize_bipolar).

    Returns:
        A new 2D array where every column has been independently normalized.
    """
    normalized_columns = [
        func(X_data[:, i]) 
        for i in range(X_data.shape[1])
    ]
    return np.column_stack(normalized_columns)
    



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
    button_multi_hot = create_final_action_events(np_data['p0_leader_pre_buttons_physical'])
    post_data_full = construct_post_data(np_data)

    # post_data_full = np.stack(
    #     [np_data[key] for key in p_consts.POST_STATE_FEATURES], axis=1
    # )

    #just need to fix the buttons, but the sticks should hopefully be discretized :)
    
    #look at hal/hal/preprocess/transformations.py for button encoding and shit
    # pre_data_full = np.stack([
    #     disc_left_stick,
    #     disc_c_stick,
    #     np_data['p0_leader_pre_triggers_physical_l'],
    #     np_data['p0_leader_pre_triggers_physical_r'],
    #     np_data['p0_leader_pre_buttons_physical']
    # ],
    #     axis=1
        
    # )
    
    pre_data_full = np.stack([  #i didnt include the physical triggers in this, but could 
        disc_left_stick,
        disc_c_stick,
        button_multi_hot
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