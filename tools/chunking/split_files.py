import os
import random
from pathlib import Path

# --- CONFIGURATION ---
# BIN_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin")
# OUTPUT_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output")

BIN_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\output\bin")
OUTPUT_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\output")

def generate_exact_splits():
    # 1. Gather exact filenames (minus the .bin extension)
    # .stem captures everything before the last dot (e.g., "00000194_101")
    file_names = [f.stem for f in BIN_DIR.glob("*.bin")]
    
    if not file_names:
        print(f"Error: No .bin files found in {BIN_DIR}")
        return

    print(f"Total .bin files detected: {len(file_names)}")

    # 2. Get user input for split ratios
    try:
        print("\n--- Split Ratio Configuration ---")
        train_p = float(input("Enter Train % (e.g., 70): "))
        val_p = float(input("Enter Val % (e.g., 15): "))
        test_p = float(input("Enter Test % (e.g., 15): "))

        if abs((train_p + val_p + test_p) - 100.0) > 1e-9:
            print(f"\n[!] Error: Ratios sum to {train_p + val_p + test_p}%. They must sum to exactly 100.")
            return
    except ValueError:
        print("\n[!] Error: Please enter valid numbers.")
        return

    # 3. Shuffle for randomness
    random.shuffle(file_names)

    # 4. Calculate slicing points
    total = len(file_names)
    train_end = int(total * (train_p / 100))
    val_end = train_end + int(total * (val_p / 100))

    # 5. Partition the data
    train_list = file_names[:train_end]
    val_list = file_names[train_end:val_end]
    test_list = file_names[val_end:]

    # 6. Save results to .txt files
    splits = {
        "t_train.txt": train_list,
        "t_val.txt": val_list,
        "t_test.txt": test_list
    }

    print("\nWriting files to output directory...")
    for filename, data in splits.items():
        file_path = OUTPUT_DIR / filename
        with open(file_path, 'w') as f:
            # Join with newline to match "line by line" requirement
            f.write('\n'.join(data))
        print(f"  -> {filename}: {len(data)} names saved.")

    print(f"\n{'='*40}")
    print(f"SUCCESS: Splits finalized.")
    print(f"Sample name written: {file_names[0] if file_names else 'N/A'}")
    print(f"{'='*40}")

if __name__ == "__main__":
    generate_exact_splits()