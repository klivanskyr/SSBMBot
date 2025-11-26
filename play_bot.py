#!/usr/bin/python3
"""
Transformer Bot - Play SSBM using a trained transformer model

This script loads a trained model and uses it to control a character in real-time.

The bot automatically sets up a CPU opponent by using two controllers:
- Controller 1 (bot): Controlled by the transformer model
- Controller 2 (opponent): Configured as CPU, then disconnected after setup

Usage:
    ./run_transformer_bot.sh
    
    OR
    
    python play_bot.py --checkpoint checkpoints/best_model.pt --character FOX
"""
import argparse
import signal
import sys
import melee
import torch
import numpy as np
from collections import deque
from pathlib import Path

from model.model import MarthTransformer
from dataset.discretizer import ActionDiscretizer
import p_consts


class GameStateProcessor:
    """Convert libmelee GameState to model input format."""
    
    def __init__(self, bot_port, opponent_port):
        self.bot_port = bot_port
        self.opponent_port = opponent_port
        self.state_buffer = deque(maxlen=16)  # Store last 16 frames
        
    def extract_state_features(self, gamestate):
        """
        Extract 48 features from gamestate for both players.
        Features match POST_STATE_FEATURES from p_consts.py
        
        Returns: numpy array of shape [48]
        """
        bot_player = gamestate.players[self.bot_port]
        opp_player = gamestate.players[self.opponent_port]
        
        features = []
        
        # Bot player features (24 features)
        features.extend([
            float(not bot_player.on_ground if bot_player.on_ground is not None else 0),  # airborne = not on_ground
            float(bot_player.action.value if bot_player.action else 0),
            float(bot_player.character.value if bot_player.character else 0),
            0.0,  # combo_count - not available in libmelee
            float(bot_player.facing if bot_player.facing is not None else 1),  # facing is a boolean
            float(bot_player.on_ground if bot_player.on_ground is not None else 0),
            float(bot_player.hitlag_left if bot_player.hitlag_left is not None else 0),
            float(bot_player.invulnerability_left if bot_player.invulnerability_left is not None else 0),
            float(bot_player.jumps_left if bot_player.jumps_left is not None else 0),
            float(bot_player.is_powershield if bot_player.is_powershield is not None else 0),
            0.0,  # last_attack_landed - not available
            0.0,  # last_hit_by - not available
            float(bot_player.percent if bot_player.percent is not None else 0),
            float(bot_player.position.x if bot_player.position else 0),
            float(bot_player.position.y if bot_player.position else 0),
            float(bot_player.shield_strength if bot_player.shield_strength is not None else 60),
            float(bot_player.action.value if bot_player.action else 0),  # action state
            float(bot_player.action_frame if bot_player.action_frame is not None else 0),  # state_age
            float(bot_player.stock if bot_player.stock is not None else 0),
            float(bot_player.speed_x_attack if bot_player.speed_x_attack is not None else 0),
            float(bot_player.speed_y_attack if bot_player.speed_y_attack is not None else 0),
            float(bot_player.speed_air_x_self if bot_player.speed_air_x_self is not None else 0),
            float(bot_player.speed_ground_x_self if bot_player.speed_ground_x_self is not None else 0),
            float(bot_player.speed_y_self if bot_player.speed_y_self is not None else 0),
        ])
        
        # Opponent player features (24 features)
        features.extend([
            float(not opp_player.on_ground if opp_player.on_ground is not None else 0),  # airborne = not on_ground
            float(opp_player.action.value if opp_player.action else 0),
            float(opp_player.character.value if opp_player.character else 0),
            0.0,  # combo_count - not available
            float(opp_player.facing if opp_player.facing is not None else 1),  # facing is a boolean
            float(opp_player.on_ground if opp_player.on_ground is not None else 0),
            float(opp_player.hitlag_left if opp_player.hitlag_left is not None else 0),
            float(opp_player.invulnerability_left if opp_player.invulnerability_left is not None else 0),
            float(opp_player.jumps_left if opp_player.jumps_left is not None else 0),
            float(opp_player.is_powershield if opp_player.is_powershield is not None else 0),
            0.0,  # last_attack_landed - not available
            0.0,  # last_hit_by - not available
            float(opp_player.percent if opp_player.percent is not None else 0),
            float(opp_player.position.x if opp_player.position else 0),
            float(opp_player.position.y if opp_player.position else 0),
            float(opp_player.shield_strength if opp_player.shield_strength is not None else 60),
            float(opp_player.action.value if opp_player.action else 0),  # action state
            float(opp_player.action_frame if opp_player.action_frame is not None else 0),  # state_age
            float(opp_player.stock if opp_player.stock is not None else 0),
            float(opp_player.speed_x_attack if opp_player.speed_x_attack is not None else 0),
            float(opp_player.speed_y_attack if opp_player.speed_y_attack is not None else 0),
            float(opp_player.speed_air_x_self if opp_player.speed_air_x_self is not None else 0),
            float(opp_player.speed_ground_x_self if opp_player.speed_ground_x_self is not None else 0),
            float(opp_player.speed_y_self if opp_player.speed_y_self is not None else 0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def update(self, gamestate):
        """Add new gamestate to buffer."""
        state_features = self.extract_state_features(gamestate)
        self.state_buffer.append(state_features)
    
    def get_sequence(self):
        """
        Get sequence of states for model input.
        Pads with zeros if not enough frames yet.
        
        Returns: numpy array of shape [16, 48]
        """
        if len(self.state_buffer) == 0:
            return np.zeros((16, 48), dtype=np.float32)
        
        # Pad with first frame if we don't have 16 frames yet
        states = list(self.state_buffer)
        while len(states) < 16:
            states.insert(0, states[0])
        
        return np.array(states, dtype=np.float32)


class TransformerBot:
    """Bot that uses trained transformer model to play."""
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.discretizer = ActionDiscretizer()
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model args from checkpoint
        model_args = checkpoint.get('args', {})
        
        # Create model with saved hyperparameters
        self.model = MarthTransformer(
            state_dim=48,
            d_model=model_args.get('d_model', 256),
            nhead=model_args.get('nhead', 8),
            num_layers=model_args.get('num_layers', 4),
            dim_feedforward=model_args.get('dim_feedforward', 1024),
            dropout=model_args.get('dropout', 0.1),
            seq_len=model_args.get('seq_len', 16)
        ).to(device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded (val_loss: {checkpoint.get('val_loss', 'N/A')})")
        print(f"  Trained for {checkpoint.get('epoch', 'N/A')} epochs")
        
    def predict_action(self, state_sequence):
        """
        Predict action from state sequence.
        
        Args:
            state_sequence: numpy array [16, 48]
        
        Returns:
            Dictionary with controller inputs
        """
        # Convert to torch tensor and add batch dimension
        state_tensor = torch.from_numpy(state_sequence).unsqueeze(0).to(self.device)  # [1, 16, 48]
        
        # Get predictions from model
        with torch.no_grad():
            predictions = self.model.predict(state_tensor)
        
        # DEBUG: Print raw predictions
        print(f"Raw predictions - Joystick bin: {predictions['joystick'].item()}, "
              f"Cstick bin: {predictions['cstick'].item()}, "
              f"Buttons: {predictions['buttons'][0].cpu().numpy()}")
        
        # Convert predictions to controller inputs
        controller_inputs = self.discretizer.predictions_to_controller(predictions)
        
        return controller_inputs


def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. Must be 1, 2, 3, or 4." % value)
    return ivalue


def main():
    parser = argparse.ArgumentParser(description='Play SSBM with trained transformer bot')
    parser.add_argument('--port', '-p', type=check_port,
                        help='The controller port (1-4) your bot will play on',
                        default=1)
    parser.add_argument('--opponent', '-o', type=check_port,
                        help='The controller port (1-4) the opponent will play on',
                        default=2)
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug mode. Creates a CSV of all game states')
    parser.add_argument('--address', '-a', default="127.0.0.1",
                        help='IP address of Slippi/Wii')
    parser.add_argument('--dolphin_executable_path', '-e', default=None,
                        help='The directory where dolphin is')
    parser.add_argument('--connect_code', '-t', default="",
                        help='Direct connect code to connect to in Slippi Online')
    parser.add_argument('--iso', default=None, type=str,
                        help='Path to melee iso.')
    parser.add_argument('--checkpoint', '-c', default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--character', default='FOX',
                        help='Character for bot to play (default: FOX)')
    parser.add_argument('--stage', default='YOSHIS_STORY',
                        help='Stage to play on (default: YOSHIS_STORY)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--cpu_level', type=int, default=9,
                        help='CPU level for opponent (1-9, default: 9)')
    parser.add_argument('--opponent_character', default='FOX',
                        help='Character for CPU opponent (default: FOX)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    
    # Initialize bot
    bot = TransformerBot(args.checkpoint, device=device)
    
    # Initialize game state processor
    state_processor = GameStateProcessor(args.port, args.opponent)
    
    # Logger
    log = None
    if args.debug:
        log = melee.Logger()
    
    # Create console
    console = melee.Console(path=args.dolphin_executable_path,
                            slippi_address=args.address,
                            logger=log)
    
    # Create controllers
    controller = melee.Controller(console=console,
                                  port=args.port,
                                  type=melee.ControllerType.STANDARD)
    
    # Create a second controller to set up the CPU opponent
    # This controller will navigate menus to configure port 2 as CPU
    controller_opponent = melee.Controller(console=console,
                                           port=args.opponent,
                                           type=melee.ControllerType.STANDARD)
    
    # Signal handler
    def signal_handler(sig, frame):
        console.stop()
        if args.debug:
            log.writelog()
            print("")
            print("Log file created: " + log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run console
    console.run(iso_path=args.iso)
    
    # Connect to console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")
    
    # Connect controller
    print("Connecting controller to console...")
    if not controller.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")
    
    # Connect opponent controller (used to set up CPU)
    print("Connecting opponent controller (for CPU setup)...")
    if not controller_opponent.connect():
        print("ERROR: Failed to connect the opponent controller.")
        sys.exit(-1)
    print("Opponent controller connected")
    
    # Get character and stage enums
    character = getattr(melee.Character, args.character, melee.Character.FOX)
    stage = getattr(melee.Stage, args.stage, melee.Stage.YOSHIS_STORY)
    opponent_character = getattr(melee.Character, args.opponent_character, melee.Character.FOX)
    
    costume = 0
    frame_count = 0
    
    print(f"\n🤖 Bot ready! Playing as {character.name} on port {args.port}")
    print(f"Using device: {device}")
    print(f"Stage: {stage.name}")
    print(f"Opponent: Level {args.cpu_level} {opponent_character.name} CPU on port {args.opponent}")
    print(f"The bot will automatically set up the CPU opponent!\n")
    
    # Main game loop
    while True:
        gamestate = console.step()
        if gamestate is None:
            continue
        
        # Performance warning
        if console.processingtime * 1000 > 12:
            print(f"WARNING: Last frame took {console.processingtime*1000:.1f}ms to process.")
        
        # In-game behavior
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            frame_count += 1
            
            # Update state buffer
            state_processor.update(gamestate)
            
            # Get action from model (every frame)
            state_sequence = state_processor.get_sequence()
            action = bot.predict_action(state_sequence)
            
            # Debug: Print action every 60 frames
            if frame_count % 60 == 0 and args.port in gamestate.players:
                bot_player = gamestate.players[args.port]
                print(f"Frame {frame_count}: Action={action}, "
                      f"Pos=({bot_player.position.x:.1f}, {bot_player.position.y:.1f}), "
                      f"BotAction={bot_player.action.name if bot_player.action else 'N/A'}")
            
            # Apply action to controller
            controller.tilt_analog(melee.Button.BUTTON_MAIN, action['x'], action['y'])
            controller.tilt_analog(melee.Button.BUTTON_C, action['c_x'], action['c_y'])
            controller.press_shoulder(melee.Button.BUTTON_L, action['l_shoulder'])
            controller.press_shoulder(melee.Button.BUTTON_R, action['r_shoulder'])
            
            # Press button if needed
            if action['button'] != melee.Button.BUTTON_MAIN:
                controller.press_button(action['button'])
            else:
                controller.release_all()
            
            # Debug output every 60 frames (1 second)
            if args.debug and frame_count % 60 == 0:
                bot_player = gamestate.players[args.port]
                print(f"Frame {frame_count}: Stock={bot_player.stock}, "
                      f"Percent={bot_player.percent:.1f}, "
                      f"Action={bot_player.action.name if bot_player.action else 'N/A'}")
            
            # Log frame
            if log:
                log.logframe(gamestate)
                log.writeframe()
        
        else:
            # Menu navigation
            frame_count = 0
            state_processor.state_buffer.clear()
            
            # Check if both players are ready (bot selected character, CPU is configured)
            both_ready = False
            if gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                if args.port in gamestate.players and args.opponent in gamestate.players:
                    bot_player = gamestate.players[args.port]
                    cpu_player = gamestate.players[args.opponent]
                    if (bot_player.character == character and 
                        cpu_player.controller_status == melee.ControllerStatus.CONTROLLER_CPU and 
                        cpu_player.cpu_level == args.cpu_level and
                        cpu_player.character is not None and 
                        cpu_player.character != melee.Character.UNKNOWN_CHARACTER):
                        both_ready = True
            
            # Bot controller - select character and stage
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                controller,
                                                character,
                                                stage,
                                                args.connect_code,
                                                cpu_level=0,  # Bot is controlled, not CPU
                                                costume=costume,
                                                autostart=both_ready,  # Only start when both ready
                                                swag=False)
            
            # Opponent controller - set up as CPU
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                controller_opponent,
                                                opponent_character,
                                                stage,
                                                args.connect_code,
                                                cpu_level=args.cpu_level,  # Set to CPU mode
                                                costume=0,
                                                autostart=False,  # Let bot controller handle start
                                                swag=False)
            
            if log:
                log.skipframe()


if __name__ == "__main__":
    main()
