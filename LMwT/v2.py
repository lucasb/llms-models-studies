# v2.py
# based on this colab: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=wJpXpmjEYC_T

import torch
import torch.nn as nn
from torch.nn import functional as F

# ###########################################
# Hyperparams
# ##########################################
batch_size = 32  # independent sequences to process in parallel
block_size = 8  # known as context length for prediction
max_iters = 5000  # max of loop steps in train
eval_interval = 300
learning_rate = 1e-3  # size of grad updates, impact the velocity of learning
device = "cpu"  # Nvidea replace with 'cuda' if intel 'xpu'
eval_iters = 200  # max loop steps in evaluation mode
num_embed_dim = 32  # numeber of embedding dimentions

# keep the rand fucntion fixed.
torch.manual_seed(1337)

# read the dataset of shakespeare texts
with open("input.txt", "r", encoding="UTF-8") as f:
    text = f.read()

# descovery our vocabolary is
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ###########################################
# Tokenize (Encoding and Decoding chars)
# strategy: use the position on chars list as the reference for enconding and decoding
# ###########################################

# create a dictionary mapping from char to int and vice versa
char2int = {char: idx for idx, char in enumerate(chars)}
int2char = {idx: char for idx, char in enumerate(chars)}


# HELPERS: encode and decode functions
def encode(str):
    return [char2int[char] for char in str]


def decode(int):
    return "".join([int2char[idx] for idx in int])


# ############################################
# Enconding the dataset and storing it into a Tensor
# ############################################
data = torch.tensor(encode(text), dtype=torch.long)

# ###########################################
# Let's split up the data into train and validation sets
# ###########################################

# Traning data is the first 90% of the dataset and validation the rest ~10%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ##########################################
# Data loading in parallel with batch  process
# ##########################################


# function to get batch of context and tatget based on block_size
def get_batch(split):
    """split: string if it is train or validation"""
    dataset = train_data if split == "train" else val_data
    # return a list of rand integers with max number by dataset len - block_size in a list the size of batch
    # The removing the block_size is necessary to grantee it will not start in the end of numbers when sampling
    rand_ints = torch.randint(len(dataset) - block_size, (batch_size,))
    context = torch.stack(
        [dataset[rand_int : rand_int + block_size] for rand_int in rand_ints]
    )
    context = context.to(device)
    target = torch.stack(
        [dataset[rand_int + 1 : rand_int + block_size + 1] for rand_int in rand_ints]
    )
    target = target.to(device)
    return context, target


# ##################################################
# MODEL: implement model archteture for bigrams
# ##################################################


# implement the function to estimate loss of training and validation in average by batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # move the model to evaluation mode
    for split in ["train", "val"]:  # eval by dataset splits
        losses = torch.zeros(eval_iters)  # tensor to calculate losses
        for k in range(eval_iters):
            ctx, trgt = get_batch(split)  # get batch tensor context and targets
            print(ctx, trgt)
            _, loss = model(
                ctx, trgt
            )  # get loss for the model at the current iteration
            losses[k] = loss.item()  # save it in losses tensor
        out[split] = losses.mean()  # calculate the mean loss of the current iteration
    model.train()  # move the model back to training mode
    return out


# impelent the single head of self-attention
class Head(nn.Module):  # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        # start the attention nodes
        self.key = nn.Linear(num_embed_dim, head_size, bias=False)
        self.query = nn.Linear(num_embed_dim, head_size, bias=False)
        self.value = nn.Linear(num_embed_dim, head_size, bias=False)
        # register the tril to use in mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, embedded):
        B, T, C = embedded.shape
        k = self.key(embedded)  # (B, T, C)
        q = self.query(embedded)  # (B, T, C)
        # compute attention scores ("affinities")
        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # why traspose? multiply by C to nomalize values for softmax
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(embedded)
        out = weights @ v
        return out


# implement the Module in neural network from PyTorch
class BigramsLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # here each token direcly read off the logits from the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, num_embed_dim)
        self.self_attention_head = Head(num_embed_dim)
        self.lang_model_head = nn.Linear(num_embed_dim, vocab_size)

    def forward(self, context, targets=None):
        T, B = context.shape  # save the context dimention in T

        # it return the probabilities on the table compering prob of each char be follow by another
        tokens_embedded = self.token_embedding_table(
            context
        )  # Batch(batch_size), Time(block_size), Channel(vocab_size)
        position_embedded = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        embedded = tokens_embedded + position_embedded  # (B, T, C)
        embedded = self.self_attention_head(embedded)
        logits = self.lang_model_head(embedded)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # decompose the logits by each variable
            logits = logits.view(
                B * T, C
            )  # reorganiza to tesor concatenating batch_size * block_size and keep C: 2D tensor
            targets = targets.view(
                B * T
            )  # rearrange the target into 1D tenor, THe B and T is the same of the logits base on get_batch fucntion return
            # calculate the loss on logitsvs targets
            loss = F.cross_entropy(
                logits, targets
            )  # ideal initial loss is -ln(1/65) = 4.1743872699

        return logits, loss

    def generate(self, context, max_new_tokens):
        # context is [B (batch_size), T(block_size)] of indexes of the current context
        for _ in range(max_new_tokens):
            # crop the context to the lest block_size tokens
            context_crop = context[:, -block_size:]
            # get the current prediction
            logits, _ = self(context_crop)  # ignoring the loss tha should be None here
            # focus only on the last time step (T)
            logits = logits[:, -1, :]  # it become (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # get a sample from the distribution
            ctx_next = torch.multinomial(
                probs, num_samples=1
            )  # reduce it only to B dementional (B, 1)
            # append sampled index to the running sequence
            context = torch.cat(
                (context, ctx_next), dim=1
            )  # keep B and add 1 to T (B, T+1)

        return context


model = BigramsLanguageModel()
m = model.to(device)

# #############################################
# Training the model
# #############################################

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# staps for Traning
for iter in range(max_iters):
    # evaluate the loss on training and validation dataset based on eval intervals
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: training loss {losses['train']:.4f} val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    context, targets = get_batch("train")

    # evaluate the loss
    logits, loss = m(context, targets)
    # initialize optimizer with zero grad
    optimizer.zero_grad(set_to_none=True)
    # calc loss
    loss.backward()
    optimizer.step()


# ############################################
# Generate from the model
# ############################################
def generate_sample(max_tokens=500):
    # generate samples
    init_context_index = torch.zeros(
        (1, 1), dtype=torch.long, device=device
    )  # generate a tensor = [[0]] (B, T) or [0][0] = 0
    generated_tokens = m.generate(
        init_context_index, max_tokens
    )  # generate a list of tokens [[0, 1, 2, 3, ..., 99]] (B, T)
    text_decoded = decode(
        generated_tokens[0].tolist()
    )  # remove the first dimention of tokens to [0, 1,2, 3, ..., 99] and convert form tensor to a list
    print(text_decoded)


generate_sample()

# stoped in 1h20min
