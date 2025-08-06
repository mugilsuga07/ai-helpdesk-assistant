# Implementing Transformer from Scratch: Your AI Helpdesk Assistant

This project builds a GPT-style Transformer model from scratch using PyTorch and trains it on realistic IT support queries (e.g., password resets, VPN access). It features a clean Streamlit UI for demo.

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
git clone https://github.com/YOUR_USERNAME/transformer-from-scratch.git
cd transformer-from-scratch
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
