import argparse
import sys
from pathlib import Path

def extract_base_id(filename):
    # E.g., '00000003_thread_v2_103' -> '00000003'
    return filename.split('_')[0]

def check_leakage(train_list_path, test_list_path):
    train_list_path = Path(train_list_path)
    test_list_path = Path(test_list_path)
    
    if not train_list_path.is_file():
        print(f"Error: Could not find train list at {train_list_path}")
        sys.exit(1)
        
    if not test_list_path.is_file():
        print(f"Error: Could not find test list at {test_list_path}")
        sys.exit(1)

    print(f"Loading {train_list_path.name}...")
    with open(train_list_path, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f if line.strip()]
        
    print(f"Loading {test_list_path.name}...")
    with open(test_list_path, 'r', encoding='utf-8') as f:
        test_files = [line.strip() for line in f if line.strip()]
        
    # Extract the base IDs
    train_bases = set(extract_base_id(f) for f in train_files)
    test_bases = set(extract_base_id(f) for f in test_files)
    
    # Calculate intersection
    overlap = train_bases.intersection(test_bases)
    
    print("\n" + "="*50)
    print("DATA LEAKAGE REPORT")
    print("="*50)
    print(f"Total entries in train list: {len(train_files):,}")
    print(f"Total entries in test list:  {len(test_files):,}")
    print(f"Unique base parts in train:  {len(train_bases):,}")
    print(f"Unique base parts in test:   {len(test_bases):,}")
    print("-"*50)
    
    if len(overlap) > 0:
        print(f"⚠️  WARNING: DATA LEAKAGE DETECTED! ⚠️")
        print(f"There are {len(overlap):,} base parts that appear in BOTH train and test sets!")
        print(f"This represents {len(overlap) / len(test_bases) * 100:.2f}% of your unique test set.")
        print(f"This means the model has seen topological variations of these test parts during training.")
        
        print("\nSample overlapping base IDs:")
        for idx, base_id in enumerate(list(overlap)[:15]):
            print(f"  - {base_id}")
    else:
        print(f"✅ SUCCESS: No data leakage detected! The splits are 100% clean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for data leakage between train and test splits.")
    parser.add_argument("--train_txt", type=str, required=True, help="Path to the train.txt split file")
    parser.add_argument("--test_txt", type=str, required=True, help="Path to the test.txt split file")
    
    args = parser.parse_args()
    check_leakage(args.train_txt, args.test_txt)
