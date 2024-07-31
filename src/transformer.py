import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long) 
    return ptt
# -----------------------------------------------------------------------------
class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
    def forward(self, idx, targets=None):
        B,T = idx.size()
        assert T<= self.config.block_size, f"cannot forward sequence of length {T}, block size {B}"

        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings(T, n_embd)
        tok_emb = self.transformer.wte(idx) # token/meaning embedding(B,T, n_embd)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        assert model_type in {'openai-community/gpt2', 'openai-community/gpt2-medium', 'openai-community/gpt2-large', 'openai-community/gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'openai-community/gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'openai-community/gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'openai-community/gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'openai-community/gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"applying decay to {len(decay_params)} tensors with {num_decay_params}")
        print(f"non decayed parameter tensors {len(nodecay_params)} with {num_nodecay_params}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # data_root = "dolly_dataset"
        data_root = "wiki"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        # print(shards)
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"init dataloader, shards legnth {len(shards)} , shard size is 20k")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            print("pull next batch: ", x.shape, y.shape)
        return x, y

#  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# main
warmup_steps = 100
max_steps = 5000
max_lr = 3e-4
min_lr = max_lr * 0.1

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) /warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps) /(max_steps-warmup_steps)
    assert 0<= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff *(max_lr - min_lr)
     

if __name__ == "__main__":
    device = "cuda"
    return_seq = 5
    max_len = 30

    model = GPT(GPTConfig(vocab_size=50304)) # init with random weights
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')

    total_batch_size = 524288 # 512k tokens, 2**19  
    # total_batch_size =
    B = 32 # microbatch size
    T = 1024 # sequence
    assert total_batch_size % (B*T) == 0, "total batch needs to be divisble by B and T"
    grad_accum_steps = total_batch_size//(B*T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"calcualted gradient accumlation steps:{grad_accum_steps}")

    # only one gpu, hard code processrank and num_processes
    train_loader = DataLoaderLite(B=B,T=T,process_rank=0, num_processes=1, split="train")
    torch.set_float32_matmul_precision('high')

    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
    # new optim model allows for weight decay (diff from lr decay)

    for i in range(max_steps):
        t0 = time.time()
        loss_accum = 0.0
        optim.zero_grad()
        
        for micro_step in range(grad_accum_steps):
            x,y = train_loader.next_batch()
            x,y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x,y)
            loss = loss/grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0) * 1000
        optim.step()
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed/dt 
        print(f"step {i} loss = {loss.item():.6f} time = {(t1 - t0)*1000:.2f} norm = {norm:.3f} m/s and tokens per second {tokens_per_sec:.2f} loss accum = {loss_accum.item():.5f}")

    # save weights after training
    torch.save(model.state_dict(), 'run2.pth')
