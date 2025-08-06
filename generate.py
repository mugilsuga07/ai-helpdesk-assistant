import torch
from model.gpt import GPT
from utils.dataset import CharDataset

with open("data/sample.txt", "r") as f:
    raw_text = f.read()

context_length = 64
dataset = CharDataset(raw_text, context_length)
stoi = dataset.stoi
itos = dataset.itos
vocab_size = dataset.vocab_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(
    vocab_size=vocab_size,
    context_length=context_length,
    model_dim=128,
    num_heads=4,
    num_layers=4
)
model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
model.to(device)
model.eval()

def generate(model, prompt, max_new_tokens):
    context = torch.tensor(
        [stoi[c] for c in prompt if c in stoi], dtype=torch.long, device=device
    ).unsqueeze(0)
    for _ in range(max_new_tokens):
        context_cond = context[:, -context_length:]
        logits = model(context_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)
        out_text = ''.join([itos[i] for i in context[0].tolist()])
        if "Customer:" in out_text[len(prompt):]:
            break
    return out_text[len(prompt):].strip().split("Customer:")[0].strip()

prompt = input("Enter customer query: ")
formatted_prompt = f"Customer: {prompt.strip()}\nAgent:"
output = generate(model, formatted_prompt, 120)

if output.startswith("Agent:"):
    print(output.strip())
else:
    print("Agent:", output.strip())





