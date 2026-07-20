import argparse
import json
import random
import sys
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, jaccard_score

# Add current path to sys.path so we can import internal modules
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.inference.json_to_brepmfr_pyg_optimized import tensors_from_brep_json_dict
from scripts.inference.run_pyg_inference import load_brepseg_for_inference, predict_probs_per_node, _batch_to_device, FACE_LABEL_NAME
from data.collator import collator

def process_unseen_data(json_dir, checkpoint_path, max_samples=10000, device='cuda', num_classes=25, inference_profile='lite', workers=0, map_json=None, train_txt=None, test_txt=None):
    json_dir = Path(json_dir)
    checkpoint_path = Path(checkpoint_path)
    
    if not json_dir.is_dir():
        print(f"Error: JSON directory {json_dir} does not exist.")
        return
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint {checkpoint_path} does not exist.")
        return

    # 0. Load Training Base IDs to guarantee strict unseen status
    train_base_ids = set()
    if train_txt:
        train_path = Path(train_txt)
        if train_path.is_file():
            print(f"Loading training split from {train_path} to exclude overlapping base parts...")
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Extract base ID (e.g. '00059048_both_v4_102' -> '00059048')
                        train_base_ids.add(line.split('_')[0])
            print(f"Found {len(train_base_ids):,} unique base IDs in training set.")
        else:
            print(f"Warning: train_txt {train_path} not found. Proceeding without strict base-part exclusion.")

    # 1. Gather and sample JSON files
    all_json_files = []
    
    if test_txt:
        test_path = Path(test_txt)
        if test_path.is_file():
            print(f"Loading pre-sampled test files from {test_path}...")
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Reconstruct full path (assuming line is either a full filename '123.json' or just the stem '123')
                        file_name = line if line.endswith('.json') else f"{line}.json"
                        full_path = json_dir / file_name
                        if full_path.is_file():
                            all_json_files.append(full_path)
            print(f"Found {len(all_json_files):,} valid files from {test_path.name}. Skipping directory scan.")
            json_files = all_json_files
        else:
            print(f"Error: test_txt {test_path} not found.")
            return
    else:
        print(f"Scanning for JSON files in {json_dir}...")
        all_json_files = list(json_dir.rglob("*.json"))
        print(f"Found {len(all_json_files):,} total JSON files.")
        
        # Filter out files whose base ID is in train_base_ids
        if train_base_ids:
            filtered_files = []
            for jf in all_json_files:
                base_id = jf.stem.split('_')[0]
                if base_id not in train_base_ids:
                    filtered_files.append(jf)
            print(f"Filtered out {len(all_json_files) - len(filtered_files):,} files overlapping with training set.")
            all_json_files = filtered_files
            print(f"Remaining strictly unseen files: {len(all_json_files):,}")
    
        if len(all_json_files) == 0:
            print("No strictly unseen JSON files found. Exiting.")
            return
            
        if len(all_json_files) > max_samples:
            print(f"Sampling {max_samples:,} strictly unseen files...")
            json_files = random.sample(all_json_files, max_samples)
        else:
            json_files = all_json_files
    
        # 1.1 Export the sampled filenames so we can check for leakage later
        sampled_list_path = Path("unseen_sampled_files.txt")
        with open(sampled_list_path, 'w', encoding='utf-8') as f:
            for jf in json_files:
                f.write(f"{jf.stem}\n")
        print(f"Saved the list of sampled files to: {sampled_list_path.resolve()}")

    # 1.5. Load Label Map
    label_map = None
    if map_json:
        map_path = Path(map_json)
        if map_path.is_file():
            print(f"Loading label map from {map_path}...")
            with open(map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
        else:
            print(f"Warning: Label map file {map_path} not found. Skipping remapping.")

    # 2. Load the model
    print(f"Loading checkpoint from {checkpoint_path}...")
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model, loaded_num_classes = load_brepseg_for_inference(checkpoint_path, device_obj, max_nodes_for_a3=768)
    model.eval()
    print(f"Model loaded successfully on {device_obj}. Checkpoint classes: {loaded_num_classes}")

    # Ensure num_classes matches the model
    num_classes = loaded_num_classes

    all_preds = []
    all_gts = []
    
    success_count = 0
    fail_count = 0

    print("Starting conversion and inference loop...")
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            # 3. Read JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 3.5 Remap Face Labels (if map is provided)
            if label_map:
                for face in data.get("faces", []):
                    # Get raw label (as string for map matching)
                    lbl_raw = str(face.get("label", "0"))
                    if lbl_raw in label_map:
                        face["label"] = label_map[lbl_raw]
                        
            # 4. Convert to PyG Graph
            pyg_graph, labels_list = tensors_from_brep_json_dict(
                data,
                inference_profile=inference_profile,
                shortest_path_workers=workers
            )

            if pyg_graph.node_data.size(0) == 0:
                fail_count += 1
                continue

            # 5. Collate into batch
            batch = collator([pyg_graph], multi_hop_max_dist=16, spatial_pos_max=32)
            batch = _batch_to_device(batch, device_obj)

            # 6. Inference
            with torch.no_grad():
                probs = predict_probs_per_node(model, batch, num_classes)
                
            probs_np = probs.cpu().numpy()
            preds = probs_np.argmax(axis=1)
            gt = pyg_graph.label_feature.cpu().numpy().flatten()
            
            # Accumulate metrics
            all_preds.extend(preds.tolist())
            all_gts.extend(gt.tolist())
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            # traceback.print_exc()
            continue
            
    print("\n--- Processing Summary ---")
    print(f"Total processed successfully: {success_count}")
    print(f"Total failed (parsing/conversion errors): {fail_count}")
    
    if len(all_preds) == 0:
        print("No predictions were made. Exiting.")
        return

    # 7. Print metrics
    unique_classes = sorted(list(set(all_gts) | set(all_preds)))
    
    if num_classes == 3:
        custom_names = {0: "Stock", 1: "Thread", 2: "Text"}
        target_names = [custom_names.get(i, f"Class {i}") for i in unique_classes]
    else:
        target_names = [FACE_LABEL_NAME.get(i, f"Class {i}") for i in unique_classes]
    
    acc = accuracy_score(all_gts, all_preds)
    
    report_dict = classification_report(
        all_gts, 
        all_preds, 
        labels=unique_classes,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    iou = jaccard_score(all_gts, all_preds, average='macro')
    per_class_accuracy = report_dict['macro avg']['recall']
    
    print("\n--- Metrics on Unseen Data ---")
    print(f"per_face_accuracy: {acc}")
    
    for i, class_name in enumerate(target_names):
        if class_name in report_dict:
            print(f"class_{i+1}_acc: {report_dict[class_name]['recall']}")
            
    print(f"per_class_accuracy: {per_class_accuracy}")
    print(f"IoU: {iou}\n")
    
    print(classification_report(
        all_gts, 
        all_preds, 
        labels=unique_classes,
        target_names=target_names,
        zero_division=0,
        digits=4
    ))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate unseen JSON files on a trained checkpoint.")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to the directory containing unseen JSON files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt model")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of files to sample (default: 10000)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--inference_profile", type=str, default="lite", choices=["lite", "no_a2", "full"], 
                        help="Profile for PyG conversion. Use 'lite' for fastest processing if the model does not require A1/A3.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for shortest path BFS if using no_a2/full")
    parser.add_argument("--map_json", type=str, default=None, help="Path to a JSON file for remapping face labels before inference (e.g. scripts/threads/remap_maps/thread_text_sw_to_brep.json)")
    parser.add_argument("--train_txt", type=str, default=None, help="Path to the train.txt split file to exclude any overlapping base geometries from the test")
    parser.add_argument("--test_txt", type=str, default=None, help="Path to a pre-sampled list of test files (like unseen_sampled_files.txt) to skip directory scanning")
    
    args = parser.parse_args()
    process_unseen_data(
        args.json_dir, 
        args.checkpoint, 
        args.max_samples, 
        args.device, 
        inference_profile=args.inference_profile, 
        workers=args.workers,
        map_json=args.map_json,
        train_txt=args.train_txt,
        test_txt=args.test_txt
    )
