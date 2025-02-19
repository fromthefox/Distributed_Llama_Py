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

def concat_tensors(tensor_list: list, dim: int) -> torch.Tensor:
    if not tensor_list:
        raise ValueError("tensor_list == None!")
    
    return torch.cat(tensor_list, dim=dim)

def split_matrix(matrix, ratios_list, dim):
    """
    Split the matrix along specified dimension and return the chunks.
    matrix: raw matrix, tensor, n dims
    ratios_list: the split ratio of matrix, list
    dim: the dimension to split (0-based index)
    """
    pieces_num = sum(ratios_list)
    shape_dim = matrix.shape[dim]
    
    if shape_dim % pieces_num != 0:
        raise ValueError(f"Dimension {dim} ({shape_dim}) not divisible by {pieces_num}")
    
    piece_size = shape_dim // pieces_num
    split_tuple = tuple(i * piece_size for i in ratios_list)
    
    # Validate split_tuple
    if sum(split_tuple) != shape_dim:
        raise ValueError(f"split_tuple {split_tuple} sum does not match shape dimension {shape_dim}")
    
    chunks = torch.split(tensor=matrix, split_size_or_sections=split_tuple, dim=dim)
    

    return chunks