import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import numpy as np


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenizes the entire text
        token_ids = tokenizer.encode(text)

        # Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns a single row from the dataset
    def __getitem__(self, idx):        
        return self.input_ids[idx], self.target_ids[idx]



def CreateDataLoader_V1(text, batch_size=4, max_length=256, stride=128, 
                        shuffle=True, drop_last=True, num_workers=0):

    token_ids = tokenizer.encode(text)

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, 
                            batch_size= batch_size,
                            shuffle= shuffle,
                            drop_last= drop_last,
                            num_workers= num_workers
                           )
    return dataloader
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  ## Size of the subspace each attention head operates on

        self.W_query = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias= qkv_bias) 
        self.out_proj = nn.Linear(d_out, d_out)  # This layer mixes information across different attention heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                             "mask",
                              torch.triu(torch.ones(context_length, context_length),
                              diagonal=1)
                            )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)      
        queries = self.W_query(x)   
        values = self.W_value(x)

        
        # Divide the embedding dimension across attention heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)         
        queries = queries.transpose(1, 2)   
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]   
        attn_scores.masked_fill_(mask_bool, -torch.inf)    
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  
        context_vec = context_vec.contiguous().view(
                                b, num_tokens, self.d_out
                                                    )
        context_vec = self.out_proj(context_vec)   
        return context_vec



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var - self.eps)
        return self.scale * norm_x + self.shift



class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
        (x + 0.044715 * torch.pow(x, 3))
                                        ))


        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)



        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
             d_in=cfg["emb_dim"],
             d_out=cfg["emb_dim"],
             context_length=cfg["context_length"],
             num_heads=cfg["n_heads"], 
             dropout=cfg["drop_rate"],
             qkv_bias=cfg["qkv_bias"]
                                    )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x




class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
                                       )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
                                 )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits




def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):

        # Keep only the last `context_size` tokens as model input
        idx_cond = idx[:, -context_size:]

        # Disable gradient tracking (inference mode)
        with torch.no_grad():
            # Forward pass through the model
            # Output shape: (batch_size, n_tokens, vocab_size)
            logits = model(idx_cond)

        # Take logits from the last time step
        # Shape: (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # Convert logits to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Select the token with the highest probability (greedy decoding)
        # Shape: (batch_size, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append the new token to the existing sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx




def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())




# This prevents silent errors and ensures correct weight mapping
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


    



def load_weights_into_gpt(gpt, params):
    # 1) Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # 2) Transformer blocks
    for b in range(len(params["blocks"])):

        # ---- Attention: QKV (GPT-2 stores them concatenated) ----
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        # ---- Attention output projection ----
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # ---- Feedforward (MLP) ----
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # ---- LayerNorms inside block ----
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    # 3) Final LayerNorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # 4) Output head (weight tying with token embeddings)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])





def train_model_simple(
    model, train_loader, val_loader, 
    optimizer, device, num_epochs, 
    eval_freq, eval_iter, start_context, tokenizer):

    # Track metrics over time
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    # Main training loop: epoch -> batches
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()
            optimizer.step()

            token_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # Optional: qualitative check (sample generation)
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen 



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # switch to evaluation mode

    with torch.no_grad():  # disable gradient tracking
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()  # switch back to training mode
    return train_loss, val_loss
    


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)        
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
           logits.flatten(0, 1), 
           target_batch.flatten()
    )
    return loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))

    model.train()


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    # if the data loader is empty, return NaN
    if len(data_loader) == 0:
        return NaN

    
    # If num_batches is not specified, use all batches
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))


    # Iterate over the dataloader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )

        total_loss += loss.item()

        return total_loss / num_batches



def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    model: language model
    idx: (batch, seq_len) token ids
    max_new_tokens: how many tokens to generate
    context_size: max context window length
    temperature: 0.0 => greedy (argmax), >0 => sampling
    top_k: if set, restrict sampling to top-k tokens
    eos_id: if set, stop when EOS is generated (works best for batch=1)
    """
    for _ in range(max_new_tokens):
        # 1) Keep only the last context_size tokens
        idx_cond = idx[:, -context_size:]

        # 2) Forward pass (no gradients during inference)
        with torch.no_grad():
            logits = model(idx_cond)              # (batch, seq, vocab)

        # 3) Only last position logits are used for next-token selection
        logits = logits[:, -1, :]                 # (batch, vocab)

        # 4) Top-k masking (optional)
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k, dim=-1)   # (batch, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)           # (batch, 1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=logits.device),
                logits
            )

        # 5) Choose next token: sampling (temperature>0) or greedy (temperature==0)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 6) Early stop on EOS (practical for batch=1; for batch>1 needs per-row handling)
        if eos_id is not None:
            if (idx_next == eos_id).all():
                break

        # 7) Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()                  
    ax2.plot(tokens_seen, train_losses, alpha=0)    
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()