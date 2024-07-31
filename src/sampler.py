import torch
import torch.nn as nn
import tiktoken
from torchinfo import summary
import torch.nn.functional as F

from transformer import CausalSelfAttention, GPTConfig, GPT 

path = "trainingRuns/first_run.pth"

former = GPT(GPTConfig(vocab_size=50304))
weights = torch.load(path)
# load state/weights now

former.load_state_dict(weights)
enc = tiktoken.get_encoding('gpt2')
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#sample from model
former.eval()
tokens = enc.encode("Hello I am a large language model ")
tokens = torch.tensor(tokens, dtype=torch.long)
former = former.to(device)


# batch/responses = 10, generate 10 different responses based on sampling
tokens = tokens.unsqueeze(0).repeat(5,1) # 5 tokens -> 5 responses, 8 tokens each
x = tokens.to(device)
x.shape


max_len = 10
while x.size(1) < max_len:
    with torch.no_grad():
        logits,loss = former(x) 

        # only get last location logits
        logits= logits[:, -1, :]

        probs = F.softmax(logits,dim=1)
        
        # top 50 samples(50 is default)
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indicies, -1, ix)

        x = torch.cat((x,xcol), dim=1)


        # diff prompts
        for i in range(5):
            tokens = x[i, :max_len].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)
 