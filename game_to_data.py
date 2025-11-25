import peppi_py
import pyarrow as pa
import numpy as np

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






#these are all the included features, can be changed 
post_state_features = [
    'p0_leader_post_airborne',
    'p0_leader_post_animation_index',
    'p0_leader_post_character',
    'p0_leader_post_combo_count',
    'p0_leader_post_direction',
    'p0_leader_post_ground',
    'p0_leader_post_hitlag',
    'p0_leader_post_hurtbox_state',
    'p0_leader_post_jumps',
    'p0_leader_post_l_cancel',
    'p0_leader_post_last_attack_landed',
    'p0_leader_post_last_hit_by',
    'p0_leader_post_percent',
    'p0_leader_post_position_x',
    'p0_leader_post_position_y',
    'p0_leader_post_shield',
    'p0_leader_post_state',
    'p0_leader_post_state_age',
    'p0_leader_post_stocks',
    'p0_leader_post_velocities_knockback_x',
    'p0_leader_post_velocities_knockback_y',
    'p0_leader_post_velocities_self_x_air',
    'p0_leader_post_velocities_self_x_ground',
    'p0_leader_post_velocities_self_y',

    'p1_leader_post_airborne',
    'p1_leader_post_animation_index',
    'p1_leader_post_character',
    'p1_leader_post_combo_count',
    'p1_leader_post_direction',
    'p1_leader_post_ground',
    'p1_leader_post_hitlag',
    'p1_leader_post_hurtbox_state',
    'p1_leader_post_jumps',
    'p1_leader_post_l_cancel',
    'p1_leader_post_last_attack_landed',
    'p1_leader_post_last_hit_by',
    'p1_leader_post_percent',
    'p1_leader_post_position_x',
    'p1_leader_post_position_y',
    'p1_leader_post_shield',
    'p1_leader_post_state',
    'p1_leader_post_state_age',
    'p1_leader_post_stocks',
    'p1_leader_post_velocities_knockback_x',
    'p1_leader_post_velocities_knockback_y',
    'p1_leader_post_velocities_self_x_air',
    'p1_leader_post_velocities_self_x_ground',
    'p1_leader_post_velocities_self_y',
]

pre_action_features = [
    # Analog stick values (floats)
    'p0_leader_pre_joystick_x', #these still need to be discretized into one pair
    'p0_leader_pre_joystick_y', 
    'p0_leader_pre_cstick_x', #these still need to be discretized into one pair
    'p0_leader_pre_cstick_y',
    
    # Physical trigger values (floats)
    'p0_leader_pre_triggers_physical_l', 
    'p0_leader_pre_triggers_physical_r',
    
    # Physical button bitmask (integer)
    'p0_leader_pre_buttons_physical',  #this still needs to be processed
]

if __name__ == "__main__":
    # Load your game
    GAME_PATH = r"C:\Users\Eswor\OneDrive\Documents\Slippi\2025-10\Game_20251015T011631.slp"
    game = peppi_py.read_slippi(GAME_PATH)
    np_data = flatten_game_to_numpy(game)

    print(f"Flattened {len(np_data.keys())} arrays.")
    print(f"P1's X-Position array (shape): {np_data['p0_leader_post_position_x'].shape}")
    print(f"P2's Buttons array (shape): {np_data['p1_leader_pre_buttons'].shape}")

    post_data_full = np.stack(
        [np_data[key] for key in post_state_features], axis=1
    )

    pre_data_full = np.stack(
        [np_data[key] for key in pre_action_features], axis=1
    )

    print(f"\nFull Post-State array shape: {post_data_full.shape}")
    print(f"Full Pre-Action array shape: {pre_data_full.shape}")

    # S = All post-frames *except the last one*
    S = post_data_full[:-1]

    # A = All pre-frames *except the first one*
    A = pre_data_full[1:]

    print(f"\nFinal State (S) shape: {S.shape}")
    print(f"Final Action (A) shape: {A.shape}")

    print(S)

    # Now S[i] is perfectly aligned with A[i]
    # (i.e., post[i] is aligned with pre[i+1])
    # You can now save S and A to your HDF5 file.