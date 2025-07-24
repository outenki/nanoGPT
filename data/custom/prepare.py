# Prepare custom dataset for training a language model

import argparse
from typing import Any
import os
from pathlib import Path
from functools import partial

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets.load import load_dataset, load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
NUM_PROC = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
NUM_PROC_LOAD_DATASET = NUM_PROC

# Encoder
ENC = tiktoken.get_encoding("gpt2")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Dataset name to load from.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load. '
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--column-name', '-cn', dest='column_name', type=str, default='text', choices=['text', 'nonce'],
        help='Column name in the dataset to process. Default is "text".'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int,
        required=False, default=None,
        help='Limit the number of samples to process.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def load_custom_dataset(data_path: str, data_name: str, data_type: str, load_from: str) -> Dataset | DatasetDict | Any:
    # Load dataset from local path with default type
    if load_from == "local" and data_type is None:
        print(f"Loading dataset {data_path}/{data_name} from local disk...")
        full_data_path = Path(data_path) / data_name
        return load_from_disk(full_data_path)

    # Load dataset from local path with specific type
    elif load_from == "local" and data_type is not None:
        print(f"Loading dataset {data_path}/{data_name} from local disk with type {data_type}...")
        full_data_path = str(Path(data_path) / data_name)
        return load_dataset(data_type, data_files=full_data_path)

    # Load dataset from Hugging Face
    elif load_from == "hf":
        print(f"Loading dataset {data_path}/{data_name} from Hugging Face...")
        print(f"load_dataset('{data_path}', '{data_name}')")
        return load_dataset(data_path, data_name)
    else:
        raise ValueError("Invalid load_from option.")


def encode_process(example, column_name) -> dict[str, Any]:
    ids = ENC.encode_ordinary(example[column_name])  # encode_ordinary ignores any special tokens
    ids.append(ENC.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


def binary_process(dataset_dict, out_path: str):
    for split, dset in dataset_dict.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(out_path, f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            if len(batch['ids']) == 0:
                continue
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


def main():
    args = read_args()
    if not Path(args.out_path).exists():
        os.makedirs(args.out_path, exist_ok=True)
    column_name = args.column_name

    dataset = load_custom_dataset(
        data_path=args.data_path,
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )
    if type(dataset) is Dataset:
        # If a single dataset is loaded, split it to train and validation sets
        # split 10% of the dataset for validation
        splitted_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        splitted_dataset['val'] = splitted_dataset.pop('test')
    else:
        splitted_dataset = dataset

    map_function = partial(encode_process, column_name=column_name)
    # Tokenize the dataset
    tokenized = splitted_dataset.map(
        map_function,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    # Process dataset to save to binary files
    # concatenate all the ids in each dataset into one large file we can use for training
    binary_process(tokenized, args.out_path)


if __name__ == "__main__":
    main()
