# python file
# based on this colab: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=wJpXpmjEYC_T

# read file / dataset in to inspect it
with open("input.txt", "r", encoding="UTF-8") as f:
    text = f.read()

print("length of dataset in chatacters: ", len(text))
# let's take a look on the first 1k chatacters
print(text[:1000])

# descovery our vocabolary is
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

# ###########################################
# Tokenize (Encoding and Decoding chars)
# strategy: use the position on chars list as the reference for enconding and decoding
# ###########################################

# create a dictionary mapping from char to int and vice versa
char2int = {char: idx for idx, char in enumerate(chars)}
int2char = {idx: char for idx, char in enumerate(chars)}

# encode and decode functions
encode = lambda str: [char2int[char] for char in str]
decode = lambda int: "".join([int2char[idx] for idx in int])

print(encode("hii there"))
print(decode(encode("hey there")))

# ############################################
# Enconding the dataset and storing it into a Tensor
# ############################################

# import PyTorch
import torch

# encode and store in tensor
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
# lets look the encode of the first 1k chars
print(data[:1000])

# ###########################################
# Let's split up the data into train and validation sets
# ###########################################

# Traning data is the first 90% of the dataset and validation the rest ~10%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(len(train_data))
print(len(val_data))

# ##########################################
# Define the context size and batch size to parallel the process
# ##########################################

# Defining variables
torch.manual_seed(1337)  # just to keep the randint in a fix numbers
batch_size = 4  # independent sequences to process in parallel
block_size = 8  # known as context length for prediction


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
    target = torch.stack(
        [dataset[rand_int + 1 : rand_int + block_size + 1] for rand_int in rand_ints]
    )
    return context, target


ctx, trgt = get_batch("train")
print("context:")
print(ctx.shape, ctx)
print("target:")
print(trgt.shape, trgt)

# for t in range(blockV_size):
#     ctx = context[:t+1]
#     trgt = target[t]
#     print(f"when input is {ctx} the target is: {trgt}")


# ##################################################
# MODEL: implement model archteture for bigrams
# ##################################################
import torch
import torch.nn as nn
from torch.nn import functional as F

# keep the rand fucntion fixed.
torch.manual_seed(1337)


# implement the Module in neural network from PyTorch
class BigramsLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # here each token direcly read off the logits from the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, targets=None):
        # it return the probabilities on the table compering prob of each char be follow by another
        logits = self.token_embedding_table(
            context
        )  # Batch(batch_size), Time(block_size), Channel(vocab_size)

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
            # get the current prediction
            logits, _ = self(context)  # ignoring the loss tha should be None here
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


model = BigramsLanguageModel(vocab_size)

# evaluate quality
logits, loss = model(ctx, trgt)
print(logits.shape)
print(loss)


def generate_sample(max_tokens=100):
    # generate samples
    init_context_index = torch.zeros(
        (1, 1), dtype=torch.long
    )  # generate a tensor = [[0]] (B, T) or [0][0] = 0
    generated_tokens = model.generate(
        init_context_index, max_tokens
    )  # generate a list of tokens [[0, 1, 2, 3, ..., 99]] (B, T)
    text_decoded = decode(
        generated_tokens[0].tolist()
    )  # remove the first dimention of tokens to [0, 1,2, 3, ..., 99] and convert form tensor to a list
    print(text_decoded)


generate_sample()

# #############################################
# Training the model
# #############################################

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# staps for Traning
batch_size = 32  # overwrite the batch_size to more real traning

for steps in range(10000):
    # sample a batch of data
    context, targets = get_batch("train")

    # evaluate the loss
    logits, loss = model(context, targets)
    # initialize optimizer with zero grad
    optimizer.zero_grad(set_to_none=True)
    # calc loss
    loss.backward()
    optimizer.step()

print(loss.item())
generate_sample(300)

# stoped in 38min
