"""
functions of llama-3
"""
import torch
import socket_server
import socket_comm_module

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

def split_matrix(matrix: torch.Tensor, ratio_list: list, dim: int) -> tuple:
    """
    Slices a 3D tensor in a given dimension according to a given list of scales

    parameters:
        matrix: input 3D tensor (X*Y*Z).
        ratio_list: list of sliced dimensions, the sum of the elements must be equal to the length of the target dimension
        dim: the dimension to slice (0,1,2)

    Returns:
        The tuple of the split tensor
    """
    # 输入验证
    if matrix.dim() != 3:
        raise ValueError(f"The input tensor must be three-dimensional with the current dimension of{matrix.dim()}")
    
    if dim not in {0, 1, 2}:
        raise ValueError(f"The dim parameter can only be 0/1/2, which is currently{dim}")
    
    if sum(ratio_list) != matrix.size(dim):
        required = matrix.size(dim)
        actual = sum(ratio_list)
        raise ValueError(f"The sum of the split dimensions must equal the dimension{dim}（{required}≠{actual}）")
    
    if any(not isinstance(s, int) or s <= 0 for s in ratio_list):
        raise ValueError("Split sizes must all be positive integers")
    
    # 执行切分操作
    return torch.split(matrix, ratio_list, dim=dim)

def get_freqs_cis(config, tokens_length):
    """
    Get the freqs_cis tensor for the model.
    """
    zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
    freqs = 1.0 / (config.rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(tokens_length), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    
    return freqs_cis

def QKV_distribution(addrs_list:list, tar_index:int, server: socket_server.TCPServer, q_chunks:tuple, k_chunks:tuple, v_chunks:tuple) -> list:
    QKV_res_list = []

    tar_addr = addrs_list[tar_index]

    # send the layer_embedding_norm_bytes to the server
    layer_embedding_norm_bytes = socket_comm_module.pack_tensor(tensor=layer_embedding_norm_bytes)
    response_embedding = server.send_data(tar_addr, layer_embedding_norm_bytes)

    if response_embedding == b"Received": # means the client received the embedding res
        q_chunks_bytes = socket_comm_module.pack_tensor(tensor=q_chunks[tar_index])
        response_q_per_token_all_heads_piece_bytes = server.send_data(tar_addr, q_chunks_bytes)
        response_q_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_q_per_token_all_heads_piece_bytes)
        QKV_res_list.append(response_q_per_token_all_heads_piece)
        
        k_chunks_bytes = socket_comm_module.pack_tensor(tensor=k_chunks[tar_index])
        response_k_per_token_all_heads_piece_bytes = server.send_data(tar_addr, k_chunks_bytes)
        response_k_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_k_per_token_all_heads_piece_bytes)
        QKV_res_list.append(response_k_per_token_all_heads_piece)
        
        v_chunks_bytes = socket_comm_module.pack_tensor(tensor=v_chunks[tar_index])
        response_v_per_token_all_heads_piece_bytes = server.send_data(tar_addr, v_chunks_bytes)
        response_v_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_v_per_token_all_heads_piece_bytes)
        QKV_res_list.append(response_v_per_token_all_heads_piece)
    
    return QKV_res_list