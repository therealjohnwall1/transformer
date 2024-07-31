import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import multiprocess as mp

# edit as we go
# db = load_dataset("databricks/databricks-dolly-15k")
db = load_dataset("wikipedia", "20220301.en")
# print(fw.column_names)
# 2603677 tokens

local_dir = "wiki"
remote_name = "wikipediaset"
shard_size = int(100 * 5 * 16384)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] 

def tokenize(doc):
    txt = doc["title"] + doc["text"]
    # txt = doc["context"] +  doc["instruction"] + doc["response"]
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(txt))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_file(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count()//2) # use half cores in order to avoid overloading

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, db['train'], chunksize=15):
        # fit more tokens into current shard
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"wiki_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            # progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_file(filename, all_tokens_np)
            shard_index += 1
            left_over = len(tokens) - remainder
            # print(len(tokens), remainder)
            if left_over > 0:
                if left_over > shard_size:
                    print("leftovers bigger then shard size")
                    left_over = shard_size
                
                all_tokens_np[0:left_over] = tokens[remainder:remainder+left_over]
                token_count = left_over

                if progress_bar is not None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(left_over)
                 
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"wiki_{split}_{shard_index:06d}")
        print("remainder length: ", token_count)
        write_file(filename, all_tokens_np[:token_count])



