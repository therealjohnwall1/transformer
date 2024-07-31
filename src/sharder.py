# oop version of parse.py in order for more usability 
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import multiprocess as mp

wiki = load_dataset("wikipedia", "20220301.en")

class Sharder:
    def __init__(self, dataset, dataset_name, local_dir, remote_name, shard_size, enc):
        self.db = dataset
        self.db = load_dataset(dataset_name)
        self.local_dir = local_dir
        self.remote_name = remote_name
        self.shard_size = shard_size
        self.DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)
        self.enc = tiktoken.get_encoding(enc)

    # should overide for diff datasets
    def tokenize(self, doc):
        eot = self.enc._special_tokens['<|endoftext|>'] # end of text token
        txt = doc["title"] + doc["text"]
        tokens = [eot]
        tokens.extend(self.enc.encode_ordinary(txt))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    def write_file(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def shard(self):
        nprocs = max(1, os.cpu_count()//2) # use half cores in order to avoid overloading

        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            
            for tokens in pool.imap(self.tokenize, self.db['train'], chunksize=15):
                # fit more tokens into current shard
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    # if progress_bar is None:
                        # progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    # progress_bar.update(len(tokens))
                
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.DATA_CACHE_DIR, f"wiki_{split}_{shard_index:06d}")
                    remainder = self.shard_size - token_count
                    # progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    self.write_file(filename, all_tokens_np)
                    shard_index += 1
                    left_over = len(tokens) - remainder
                    print(len(tokens), remainder)
                    if left_over > 0:
                        if left_over > self.shard_size:
                            print("leftovers bigger then shard size")
                            left_over = self.shard_size
                        
                        all_tokens_np[0:left_over] = tokens[remainder:remainder+left_over]
                        token_count = left_over

                        
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.DATA_CACHE_DIR, f"wiki_{split}_{shard_index:06d}")
                print("remainder length: ", token_count)
                self.write_file(filename, all_tokens_np[:token_count])
    

if __name__ == "__main__":
    sharder = Sharder(wiki, "wikipedia", "data", "wikipedia", 2**20, "gpt2")
    sharder.shard()