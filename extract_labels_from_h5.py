# We are not using this script, because MFCAD++ labels are directly there inside step files.
# https://gitlab.com/qub_femg/machine-learning/mfcad2-dataset
# Unlike the MFCAD dataset which used Pickled Python lists saved as .face_truth files, the MFCAD++ dataset saved the class labels directly to the ADVANCED_FACES in the STEP files.

import h5py
import numpy as np

# Path to your file
h5_file_path = r'C:\Users\smr52\Desktop\MFCAD++\hierarchical_graphs\training_MFCAD++.h5'

with h5py.File(h5_file_path, 'r') as f:
    # Access the first batch
    first_batch_key = list(f.keys())[0]
    batch = f[first_batch_key]
    
    # 1. Get the model name
    model_names = batch['CAD_model'][:]
    first_model_name = model_names[0].decode('utf-8')
    
    # 2. Get the indices (the bookmarks)
    # idx[0] is the start of model 1, idx[1] is the start of model 2
    indices = batch['idx'][:].flatten() 
    
    # 3. Get all labels and slice them
    all_labels = batch['labels'][:].flatten()
    
    start_pos = int(indices[0])
    end_pos = int(indices[1])
    
    first_model_labels = all_labels[start_pos:end_pos]

    # --- OUTPUT ---
    print(f"--- Model Information ---")
    print(f"Model Name: {first_model_name}")
    print(f"Total Number of Faces: {len(first_model_labels)}")
    print(f"Face Labels: {first_model_labels.tolist()}")