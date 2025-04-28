import numpy as np
import os
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument('--pipeline_dir',    required=True, help='Directory of this specific pipeline')
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--new_sample_size", required=True, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    include_idx = np.empty(0, dtype=int)
    index_file = os.path.join(args.pipeline_dir, "index.npy")
    if args.iter > 0:
        include_idx = np.load(index_file).astype(int)
    
    train_pk = os.path.join(args.data_dir, 'data_dump_density_preprocessed_train.pk')
    with open(train_pk, 'rb') as f:
        raw_train = pickle.load(f)
    keys = list(raw_train.keys())
    num_raw_train_sample = len(keys)

    exclude_idx = np.delete(np.arange(num_raw_train_sample), include_idx)
    delete_idx = np.random.choice(exclude_idx.size, size=args.new_sample_size, replace=False)
    final_exclude_idx = np.delete(exclude_idx, delete_idx)
    final_include_idx = np.delete(np.arange(num_raw_train_sample), final_exclude_idx)

    np.save(index_file, final_include_idx)
    print(final_include_idx)

if __name__ == "__main__":
    main()
