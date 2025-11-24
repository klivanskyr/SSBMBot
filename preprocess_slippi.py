import py7zr
import os
import numpy as np
from peppi_py import read_slippi


SLP_FOLDER = "extracted/training_data"
PLAYER = 0
OPPONENT = 1


def extract_files():
    archive = "./melee_public_slp_dataset_v2.7z"
    output_folder = "extracted"

    os.makedirs(output_folder, exist_ok=True)
    print(f"Extracting {archive} to {output_folder}...")

    with py7zr.SevenZipFile(archive, 'r') as z:
        z.extractall(path=output_folder)

    print("Done!")  
    total_files = sum(len(files) for _, _, files in os.walk(output_folder))
    print(f"Total files extracted: {total_files}")

def safe_port(frame, idx):
    """Return player object safely (ports can be None or missing)."""
    try:
        port = frame.ports[idx]
        if port is None:
            return None
        return port.leader  # NEW API
    except:
        return None


def frame_to_state(frame, p_idx=0, o_idx=1):
    pass

def load_all_slp(folder):
    all_states = []

    for file in os.listdir(folder):
        if not file.endswith(".slp"):
            continue

        path = os.path.join(folder, file)
        print(f"Loading {path}")

        try:
            game = read_slippi(path)
            print(f"Game loaded: {game.metadata}")
            states = []
            # for frame in game.frames:
            #     state = frame_to_state(frame, PLAYER, OPPONENT)
            #     if state is not None:
            #         states.append(state)
            
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue
        # all_states.append(np.array(states))
    
        





if __name__ == "__main__":

    # extract_files()

    states = load_all_slp(SLP_FOLDER)
    print("Loaded states shape:", states.shape)






