import numpy as np
from dataclasses import dataclass
from typing import Tuple
import melee


@dataclass
class ActionSpace:
    JOYSTICK_BINS = 17  # neutral + 8 dirs (normal) + 8 dirs (smash/dash)
    CSTICK_BINS = 9     # neutral + 8 dirs (c-stick is effectively digital)
    TRIGGER_BINS = 3    # not pressed (0-33%), light (33-66%), full (66-100%)
    NUM_BUTTONS = 8     # A, B, X, Y, Z, L, R, Start

class ActionDiscretizer:
    """Converts continuous controller inputs to discrete action indices."""
    def __init__(self):
        self.deadzone = 0.2
        self.smash_threshold = 0.8 
        
    def discretize_stick(self, x: float, y: float, include_smash: bool = True) -> int:
        """
        Convert (x, y) stick position to discrete bin.
        
        Returns:
            0: Neutral
            1-8: Normal directions (N=1, NE=2, E=3, SE=4, S=5, SW=6, W=7, NW=8)
            9-16: Smash directions (only if include_smash=True and magnitude > threshold)
        """
        magnitude = np.sqrt(x**2 + y**2)
        
        if magnitude < self.deadzone:
            return 0  # Neutral
        
        # Calculate angle in degrees (0 = East, 90 = North)
        angle = np.degrees(np.arctan2(y, x))
        if angle < 0:
            angle += 360
        
        # Map to 8 directions (N, NE, E, SE, S, SW, W, NW)
        # Bin edges: 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5
        direction_bin = int((angle + 22.5) // 45) % 8 + 1  # 1-8
        
        # Check if this is a smash input (high magnitude)
        if include_smash and magnitude > self.smash_threshold:
            return direction_bin + 8  # 9-16 for smash directions
        
        return direction_bin  # 1-8 for normal directions
    
    def discretize_trigger(self, value: float) -> int:
        """
        Convert trigger value to discrete bin.
        
        Bins represent approximate trigger positions:
            0: Not pressed (0-33%, value < 0.33)
            1: Light press (33-66%, 0.33 <= value < 0.66) - light shield
            2: Full press (66-100%, value >= 0.66) - hard shield, digital click
        """
        if value < 0.33:
            return 0
        elif value < 0.66:
            return 1
        else:
            return 2
    
    def extract_buttons(self, bitmask: int) -> np.ndarray:
        """
        Extract individual button states from bitmask.
        
        Common SSBM button mapping (check peppi_py docs for exact values):
        Bit 0: A
        Bit 1: B
        Bit 2: X
        Bit 3: Y
        Bit 8: Z
        Bit 9: L
        Bit 10: R 
        Bit 11: Start
        
        Returns: Binary array [8] for each button
        """
        buttons = np.zeros(8, dtype=np.float32)
        buttons[0] = (bitmask >> 0) & 1   # A
        buttons[1] = (bitmask >> 1) & 1   # B
        buttons[2] = (bitmask >> 2) & 1   # X
        buttons[3] = (bitmask >> 3) & 1   # Y
        buttons[4] = (bitmask >> 8) & 1   # Z
        buttons[5] = (bitmask >> 9) & 1   # L
        buttons[6] = (bitmask >> 10) & 1  # R
        buttons[7] = (bitmask >> 11) & 1  # Start
        return buttons
    
    def discretize_action(self, action_raw: np.ndarray) -> dict:
        """
        Convert raw action array to discretized components.
        
        Args:
            action_raw: [7] array with [joy_x, joy_y, c_x, c_y, trig_l, trig_r, buttons]
        
        Returns:
            Dictionary with discretized action components
        """
        joy_x, joy_y, c_x, c_y, trig_l, trig_r, buttons_mask = action_raw
        
        return {
            'joystick': self.discretize_stick(joy_x, joy_y, include_smash=True),
            'cstick': self.discretize_stick(c_x, c_y, include_smash=False),
            'trigger_l': self.discretize_trigger(trig_l),
            'trigger_r': self.discretize_trigger(trig_r),
            'buttons': self.extract_buttons(int(buttons_mask))
        }
    
    def batch_discretize(self, actions_raw: np.ndarray) -> dict:
        """
        Discretize a batch of actions.
        
        Args:
            actions_raw: [batch_size, 7] array
        
        Returns:
            Dictionary with discretized components as arrays
        """
        batch_size = actions_raw.shape[0]
        
        joystick = np.zeros(batch_size, dtype=np.int64)
        cstick = np.zeros(batch_size, dtype=np.int64)
        trigger_l = np.zeros(batch_size, dtype=np.int64)
        trigger_r = np.zeros(batch_size, dtype=np.int64)
        buttons = np.zeros((batch_size, 8), dtype=np.float32)
        
        for i in range(batch_size):
            disc = self.discretize_action(actions_raw[i])
            joystick[i] = disc['joystick']
            cstick[i] = disc['cstick']
            trigger_l[i] = disc['trigger_l']
            trigger_r[i] = disc['trigger_r']
            buttons[i] = disc['buttons']
        
        return {
            'joystick': joystick,
            'cstick': cstick,
            'trigger_l': trigger_l,
            'trigger_r': trigger_r,
            'buttons': buttons
        }
    
    def bin_to_stick(self, bin_idx: int, include_smash: bool = True) -> Tuple[float, float]:
        """
        Convert discrete stick bin back to (x, y) coordinates.
        
        Args:
            bin_idx: Discrete bin index
            include_smash: Whether this stick supports smash inputs
        
        Returns:
            (x, y) tuple with stick coordinates
        """
        if bin_idx == 0:
            return (0.0, 0.0)  # Neutral
        
        # Check if this is a smash input
        is_smash = include_smash and bin_idx > 8
        if is_smash:
            direction_idx = bin_idx - 8  # Convert 9-16 to 1-8
            magnitude = 1.0  # Full magnitude for smash
        else:
            direction_idx = bin_idx  # 1-8
            magnitude = 0.7  # Normal tilt magnitude
        
        # Map direction index to angle
        # 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
        angle_map = {
            1: 90,   # N
            2: 45,   # NE
            3: 0,    # E
            4: 315,  # SE
            5: 270,  # S
            6: 225,  # SW
            7: 180,  # W
            8: 135,  # NW
        }
        
        angle = angle_map.get(direction_idx, 0)
        angle_rad = np.radians(angle)
        
        x = magnitude * np.cos(angle_rad)
        y = magnitude * np.sin(angle_rad)
        
        return (float(x), float(y))
    
    def bin_to_trigger(self, bin_idx: int) -> float:
        """
        Convert discrete trigger bin to analog value.
        
        Args:
            bin_idx: 0 (no press), 1 (light), or 2 (full)
        
        Returns:
            Analog trigger value 0.0-1.0
        """
        if bin_idx == 0:
            return 0.0
        elif bin_idx == 1:
            return 0.5  # Light press
        else:
            return 1.0  # Full press
    
    def predictions_to_controller(self, predictions: dict) -> dict:
        """
        Convert model predictions to controller inputs for libmelee.
        
        Args:
            predictions: Dict with model outputs (discrete indices)
                - joystick: tensor [batch] or int
                - cstick: tensor [batch] or int
                - trigger_l: tensor [batch] or int
                - trigger_r: tensor [batch] or int
                - buttons: tensor [batch, 8] or array
        
        Returns:
            Dict with controller inputs:
                - x: Main stick X (-1 to 1)
                - y: Main stick Y (-1 to 1)
                - c_x: C-stick X (-1 to 1)
                - c_y: C-stick Y (-1 to 1)
                - l_shoulder: L trigger (0 to 1)
                - r_shoulder: R trigger (0 to 1)
                - button: melee.Button enum value
        """
        
        # Extract scalar values (handle both tensor and numpy)
        joy_idx = int(predictions['joystick'][0]) if hasattr(predictions['joystick'], '__len__') else int(predictions['joystick'])
        c_idx = int(predictions['cstick'][0]) if hasattr(predictions['cstick'], '__len__') else int(predictions['cstick'])
        trig_l_idx = int(predictions['trigger_l'][0]) if hasattr(predictions['trigger_l'], '__len__') else int(predictions['trigger_l'])
        trig_r_idx = int(predictions['trigger_r'][0]) if hasattr(predictions['trigger_r'], '__len__') else int(predictions['trigger_r'])
        
        # Handle buttons (binary array)
        if hasattr(predictions['buttons'], 'cpu'):  # PyTorch tensor
            buttons_arr = predictions['buttons'][0].cpu().numpy()
        elif len(predictions['buttons'].shape) > 1:  # Batched numpy
            buttons_arr = predictions['buttons'][0]
        else:  # Already 1D
            buttons_arr = predictions['buttons']
        
        # Convert to stick positions
        joy_x, joy_y = self.bin_to_stick(joy_idx, include_smash=True)
        c_x, c_y = self.bin_to_stick(c_idx, include_smash=False)
        
        # Convert triggers
        l_shoulder = self.bin_to_trigger(trig_l_idx)
        r_shoulder = self.bin_to_trigger(trig_r_idx)
        
        # Determine which button to press (priority: A > B > X > Y > Z)
        # libmelee.Controller.simple_press only supports one button at a time
        button = melee.Button.BUTTON_MAIN  # Default (no button)
        
        if buttons_arr[0] > 0.5:  # A
            button = melee.Button.BUTTON_A
        elif buttons_arr[1] > 0.5:  # B
            button = melee.Button.BUTTON_B
        elif buttons_arr[2] > 0.5:  # X
            button = melee.Button.BUTTON_X
        elif buttons_arr[3] > 0.5:  # Y
            button = melee.Button.BUTTON_Y
        elif buttons_arr[4] > 0.5:  # Z
            button = melee.Button.BUTTON_Z
        elif buttons_arr[5] > 0.5:  # L (digital)
            button = melee.Button.BUTTON_L
        elif buttons_arr[6] > 0.5:  # R (digital)
            button = melee.Button.BUTTON_R
        
        return {
            'x': joy_x,
            'y': joy_y,
            'c_x': c_x,
            'c_y': c_y,
            'l_shoulder': l_shoulder,
            'r_shoulder': r_shoulder,
            'button': button
        }