import os
import pickle
import numpy as np
from pathlib import Path
import random
import argparse

def generate_dummy_data(n_cases, base_dir):
    # 1. Setup Directories
    # .expanduser() handles '~' and .resolve() gets the full absolute path
    base_path = Path(base_dir).expanduser().resolve()
    image_dir = base_path / "images"
    
    # Create the directory tree
    image_dir.mkdir(parents=True, exist_ok=True)
    
    case_ids = [f"case_{i:04d}" for i in range(1, n_cases + 1)]
    
    print(f"Target Directory: {base_path}")
    print(f"Generating {n_cases} numpy images...")

    # 2. Create Numpy Images
    for case_id in case_ids:
        # 512x512, float16, range 0-4096
        img_data = np.random.uniform(0, 4096, (512, 512)).astype(np.float16)
        np.save(image_dir / f"{case_id}.npy", img_data)

    # 3. Create Metadata Dictionary (ds['data'])
    print("Generating metadata dictionary...")
    ds_data = {}
    for case_id in case_ids:
        ds_data[case_id] = {
            'StudyAge': random.randint(5, 17),
            'Sex': random.choice([0, 1]),
            'Height': round(random.uniform(120, 160), 2),
            'Weight': round(random.uniform(30, 60), 2),
            'Score_Spine': round(random.uniform(0.5, 1.3), 3)
        }

    # 4. Create 5-Fold Split (ds['split'])
    print("Generating 5-fold cross-validation splits...")
    random.shuffle(case_ids)
    splits = {}
    
    fold_size = n_cases // 5
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i != 4 else n_cases
        
        valid_ids = case_ids[start:end]
        train_ids = [cid for cid in case_ids if cid not in valid_ids]
        
        splits[i] = {
            'train': train_ids,
            'valid': valid_ids
        }

    # 5. Save the Pickle File
    ds = {
        'split': splits,
        'data': ds_data
    }
    
    pickle_path = base_path / "ds.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(ds, f)

    print("-" * 30)
    print(f"Success! Data saved to: {base_path}")
    print(f"Images: {len(list(image_dir.glob('*.npy')))} files in {image_dir}")
    print(f"Pickle: {pickle_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy data.")
    
    # N Argument
    parser.add_argument(
        '-n', '--number', 
        type=int, 
        default=25, 
        help='Number of dummy cases to generate'
    )
    
    # Base Directory Argument
    parser.add_argument(
        '-d', '--dir', 
        type=str, 
        default='temp/sample_data', 
        help='Base directory to save data'
    )
    
    args = parser.parse_args()
    
    if args.number < 5:
        print("Warning: N < 5. Some CV folds will be empty.")
        
    generate_dummy_data(args.number, args.dir)