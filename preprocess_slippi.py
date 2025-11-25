import py7zr
import os
import numpy as np
import sys
import traceback
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



def frame_to_state(frame):
    pass

def safe_read_slippi(path):
    """
    Safely read a slippi file, catching all possible errors including Rust panics.
    Returns (success: bool, game: object or None, error: str or None)
    """
    try:
        # Suppress stderr temporarily to avoid panic messages flooding output
        old_stderr = sys.stderr
        try:
            # Try to read the file
            game = read_slippi(path)
            return True, game, None
        finally:
            sys.stderr = old_stderr
    except KeyboardInterrupt:
        # Re-raise keyboard interrupts
        raise
    except SystemExit:
        # Re-raise system exits
        raise
    except BaseException as e:
        # Catch everything else including PanicException
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        # Check if it's a panic
        is_panic = ("PanicException" in error_type or 
                   "panic" in error_msg.lower() or 
                   "OutOfSpec" in error_msg or 
                   "panicked" in error_msg.lower() or
                   "panic" in error_trace.lower())
        return False, None, f"{error_type}: {error_msg}" if not is_panic else "Rust panic"
    except:
       
        return False, None, "Unknown error (possibly Rust panic)"


def load_all_slp(folder):
    all_states = []
    count = 0
    skipped_count = 0

    for file in os.listdir(folder):
        if not file.endswith(".slp") :
            continue

        if file.find("Marth") == -1:
            continue

        path = os.path.join(folder, file)
        print(f"Loading {path}")

        # Use safe wrapper to read file
        success, game, error = safe_read_slippi(path)
        
        if not success:
            # Check if it's a panic-related error
            error_lower = error.lower() if error else ""
            if error and ("panic" in error_lower or "outofspec" in error_lower or "panicked" in error_lower or error == "Rust panic"):
                print(f"  -> Skipping corrupted/invalid file: Rust panic/error in peppi_py")
            else:
                print(f"  -> Could not load: {error}")
            skipped_count += 1
            # Flush output to ensure message is printed before continuing
            sys.stdout.flush()
            continue
        
        # Successfully loaded
        try:
            print(f"  -> Game loaded: {game.metadata}")
            count += 1
            states = []
            # for frame in game.frames:
            #     state = frame_to_state(frame, PLAYER, OPPONENT)
            #     if state is not None:
            #         states.append(state)
            all_states.append(np.array(states))
        except Exception as e:
            print(f"  -> Error processing game data: {e}")
            skipped_count += 1
            continue
    
    print(f"\nSummary: Loaded {count} games, skipped {skipped_count} files")
    return all_states
    
        





if __name__ == "__main__":

    # extract_files()
    states = load_all_slp(SLP_FOLDER)
    print("Loaded states shape:", states.shape)






