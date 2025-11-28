import numpy as np

#these are all the included features, can be changed 
POST_STATE_FEATURES = [
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

PRE_ACTION_FEATURES = [
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

STICK_XY_CLUSTER_CENTERS_V1: np.ndarray = np.array(
    [
        [0.0, 0.5],
        [0.04031356, 0.67579186],
        [0.04306322, 0.32213718],
        [0.07422757, 0.74398047],
        [0.08753323, 0.23118582],
        [0.09649086, 0.5],
        [0.10760637, 0.79640961],
        [0.13590235, 0.16644649],
        [0.16092533, 0.5],
        [0.16233712, 0.85917079],
        [0.1661272, 0.13657573],
        [0.2414301, 0.08239827],
        [0.25649357, 0.28935188],
        [0.26877457, 0.71068704],
        [0.27118719, 0.5],
        [0.29513732, 0.94317985],
        [0.30770162, 0.04870167],
        [0.32365757, 0.5],
        [0.5, 0.0],
        [0.5, 0.11190532],
        [0.5, 0.21006109],
        [0.5, 0.31441873],
        [0.5, 0.5],
        [0.5, 0.67654097],
        [0.5, 0.76718795],
        [0.5, 0.86723614],
        [0.5, 1.0],
        [0.67634243, 0.5],
        [0.69229841, 0.04870167],
        [0.70486271, 0.94317985],
        [0.72881281, 0.5],
        [0.73122543, 0.71068704],
        [0.74350643, 0.28935188],
        [0.7585699, 0.08239827],
        [0.8338728, 0.13657573],
        [0.83766288, 0.85917079],
        [0.83907467, 0.5],
        [0.86409765, 0.16644649],
        [0.89239365, 0.79640961],
        [0.90350914, 0.5],
        [0.91246676, 0.23118582],
        [0.92577243, 0.74398047],
        [0.95693678, 0.32213718],
        [0.95968646, 0.67579186],
        [1.0, 0.5],
    ],
    dtype=np.float32,
)

A_MASK = 256     # 0x0100
B_MASK = 512     # 0x0200
L_MASK = 64      # 0x0040
R_MASK = 32      # 0x0020
Z_MASK = 16      # 0x0010
START_MASK = 4096  # 0x1000
D_UP_MASK = 8      # 0x0008
D_DOWN_MASK = 4      # 0x0004
D_LEFT_MASK = 1      # 0x0001
D_RIGHT_MASK = 2     # 0x0002
X_MASK = 1024    # 0x0400
Y_MASK = 2048    # 0x0800

# Define the 10 final actions and their combined masks
FINAL_ACTION_MAPPING = [
    ('JUMP', X_MASK | Y_MASK),    
    ('SHIELD', L_MASK | R_MASK),
    ('A', A_MASK),
    ('B', B_MASK),
    ('Z', Z_MASK),
    ('Start', START_MASK),
    ('D_Up', D_UP_MASK),
    ('D_Down', D_DOWN_MASK),
    ('D_Left', D_LEFT_MASK),
    ('D_Right', D_RIGHT_MASK),
]

NORMALIZE_MIN_MAX = [
    'p0_leader_post_airborne',
    'p0_leader_post_animation_index',
    'p0_leader_post_character',
    'p0_leader_post_direction',
    'p0_leader_post_ground',
    'p0_leader_post_hurtbox_state',
    'p0_leader_post_jumps',
    'p0_leader_post_l_cancel',
    'p0_leader_post_last_attack_landed', #categorical
    'p0_leader_post_last_hit_by', #categorical
    'p0_leader_post_state', # categorical
    'p0_leader_post_state_age',
    'p0_leader_post_stocks',
    

    'p1_leader_post_airborne',
    'p1_leader_post_animation_index',
    'p1_leader_post_character',
    'p1_leader_post_direction',
    'p1_leader_post_ground',
    'p1_leader_post_hurtbox_state',
    'p1_leader_post_jumps',
    'p1_leader_post_l_cancel',
    'p1_leader_post_last_attack_landed', #categorical
    'p1_leader_post_last_hit_by', #categorical
    'p1_leader_post_state', # categorical
    'p1_leader_post_state_age',
    'p1_leader_post_stocks'
]

BIPOLAR_MAPPING = [
    'p0_leader_post_velocities_knockback_x',
    'p0_leader_post_velocities_knockback_y',
    'p0_leader_post_velocities_self_x_air',
    'p0_leader_post_velocities_self_x_ground',
    'p0_leader_post_velocities_self_y',
    'p0_leader_post_position_x',
    'p0_leader_post_position_y',
    
    'p1_leader_post_velocities_knockback_x',
    'p1_leader_post_velocities_knockback_y',
    'p1_leader_post_velocities_self_x_air',
    'p1_leader_post_velocities_self_x_ground',
    'p1_leader_post_velocities_self_y',
    'p1_leader_post_position_x',
    'p1_leader_post_position_y'
]

INVERSE_MAPPING = [
    'p0_leader_post_combo_count',
    'p0_leader_post_hitlag', #opp
    'p0_leader_post_percent',
    'p0_leader_post_shield',
    
    'p1_leader_post_combo_count',
    'p1_leader_post_hitlag', #opp
    'p1_leader_post_percent',
    'p1_leader_post_shield'
]