import torch
import torch.nn.functional as F
from model.gpt import GPT
from utils.dataset import CharDataset, batch_loader

context_length = 64
batch_size = 16
vocab_size = 128  
model_dim = 128
num_heads = 4
num_layers = 4
learning_rate = 1e-3
num_steps = 3000


with open("data/sample.txt", "r") as f:
    raw_text = f.read()

dataset = CharDataset(raw_text, context_length)
vocab_size = dataset.vocab_size


model = GPT(vocab_size, context_length, model_dim, num_heads, num_layers)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for step in range(num_steps):
    x, y = batch_loader(raw_text, context_length, batch_size, dataset=dataset)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")


torch.save(model.state_dict(), "gpt_model.pth")
print("Model training complete and saved to gpt_model.pth")

