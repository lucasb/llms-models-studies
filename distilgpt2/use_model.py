import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# load the model
model = torch.load('drinkGenGPT2.pt')

##
## DEFINE DEVICE AND PROCESSING
##

device = None
# models need to be attached to hardware
# if you have an NVIDIA GPU attached, use 'cuda'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    # if apple silicon, set to 'mps' - otherwise 'cpu' (not advised)
    # try:
    #     torch.mps
    #     device = torch.device('mps')
    # except:
    device = torch.device('cpu')

# the tokenizer turns texts to numbers (and vice-versa)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# the transformer
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)


def ask_a_drink(name, idx):
  print(f"##### {idx}")

  input_str = f"the ingredients for a drink called {name} are"
  input_ids = tokenizer.encode(input_str, return_tensors='pt').to(device)

  output = model.generate(
    input_ids,
    max_length=len(input_str),
    num_return_sequences=1,
    do_sample=True,
    top_k=8,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
  )

  decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
  print(decoded_output)


ask_a_drink("espresso martini", 1)
ask_a_drink("Alfonso In Wonderland", 2)
ask_a_drink("Pineapple Manhattan", 3)
ask_a_drink("Love Berry", 4)
