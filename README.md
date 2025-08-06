# Implementing Transformer from Scratch: Your AI Helpdesk Assistant

This project is a hands-on implementation of a GPT-style Transformer, built entirely from scratch using PyTorch. Drawing inspiration from the foundational paper “Attention is All You Need,” I reimplemented core transformer components—like multi-head self-attention, positional encoding, and residual connections—and adapted them to suit a real-world enterprise use case: IT support automation.

The model was trained on simulated helpdesk conversations to reflect actual workplace interactions. An assistant that can respond to queries like "My Teams isn’t loading" or "How do I reset my password?" with context-aware replies. What makes this special is that every component—from self-attention and positional encoding to training loops and text generation—was written manually, giving me a deep understanding of how large language models really work. The project also includes a simple Streamlit interface so anyone can test it out with natural queries, just like chatting with an internal IT bot.

# Features

- Custom character-level GPT-like Transformer (no Hugging Face)
- Trained on enterprise helpdesk prompts and responses
- Streamlit UI to interact with the model
- Modular codebase (attention, transformer blocks, dataset loader, training)

# Example Queries

> "I can't log in to my email"  
→ Agent: Please try resetting your password using the 'Forgot Password' link.

> "Where can I download Zoom?"  
→ Agent: Visit the Company Portal, search for Zoom, and click Install.


# Setup

```bash
git clone https://github.com/mugilsuga07/transformer-from-scratch.git
cd transformer-from-scratch
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
