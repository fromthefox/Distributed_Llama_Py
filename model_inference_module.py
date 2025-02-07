"""
functions of llama-3
"""
import torch

def input_embedding(input_text, tokenizer, config, model):
    tokens = [128000] + tokenizer.encode(input_text)
    tokens = torch.tensor(tokens)
    tokens_length = len(tokens)
    embedding_layer = torch.nn.Embedding(config.vocab_size, config.dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
    return token_embeddings_unnormalized, tokens_length

def rms_norm(tensor, norm_weights, config):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + config.norm_eps)) * norm_weights

