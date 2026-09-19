"""The engine: one encoder pass, three heads read off the same pooled state.

Nothing autoregressive happens here. The cost of a decision is one forward pass
through the encoder, not N forward passes through a decoder.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from schema import ACTIONS

# ModernBERT is bidirectional and handles 8192 tokens natively - the right default
# for classifying a long log. Qwen2.5-0.5B works too but it is causal: early tokens
# cannot see later ones, so pooling must be mean or last-token, never [CLS]/index 0.
DEFAULT_ENCODER = "answerdotai/ModernBERT-base"


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


class AgentDecisionEngine(nn.Module):
    def __init__(self, encoder_name=DEFAULT_ENCODER, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head_error = nn.Linear(d, 1)
        self.head_action = nn.Linear(d, len(ACTIONS))
        self.head_score = nn.Linear(d, 5)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.drop(masked_mean(h, attention_mask))
        return self.head_error(z).squeeze(-1), self.head_action(z), self.head_score(z)


def get_tokenizer(name=DEFAULT_ENCODER):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:                 # causal models (Qwen) ship without one
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"            # fine with mean pooling
    return tok
