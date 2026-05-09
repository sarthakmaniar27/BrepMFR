import pathlib

# Define the base directory
base_path = pathlib.Path(r"C:\Users\smr52\Desktop\MFCAD++_dataset\json")

# The sub-folders we want to process
sub_folders = ['train', 'test', 'val']

def rename_step_files():
    for folder_name in sub_folders:
        folder_path = base_path / folder_name
        
        # Check if folder exists to avoid errors
        if not folder_path.exists():
            print(f"Skipping: {folder_name} (Folder not found)")
            continue

        print(f"Processing folder: {folder_name}...")

        # Iterate over all .json files in the folder
        # Glob finds all files ending in .json
        for file_path in folder_path.glob("*.json"):
            # Construct the new name: prefix + original name
            new_name = f"mfcad_{file_path.name}"
            
            # Create the full new path
            new_file_path = file_path.with_name(new_name)
            
            # Perform the rename
            try:
                file_path.rename(new_file_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {file_path.name}: {e}")

if __name__ == "__main__":
    rename_step_files()
    print("\nTask completed!")