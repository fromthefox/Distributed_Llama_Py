"""
functions of llama-3
"""
import torch
import socket_server
import socket_comm_module
import threading


def input_embedding(input_text, tokenizer, config, model, dtype):
    tokens = [128000] + tokenizer.encode(input_text)
    tokens = torch.tensor(tokens)
    tokens_length = len(tokens)
    embedding_layer = torch.nn.Embedding(config["vocab_size"], config["dim"])
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(dtype)
    return token_embeddings_unnormalized, tokens_length

def rms_norm(tensor, norm_weights, config):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + config["norm_eps"])) * norm_weights

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
    freqs = 1.0 / (config["rope_theta"] ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(tokens_length), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    
    return freqs_cis

def QKV_distribution(addrs_list:list, ports_list:list, tar_index:int, server: socket_server.TCPServer, q_chunks:tuple, k_chunks:tuple, v_chunks:tuple, layer_embedding_norm:torch.Tensor) -> list:
    QKV_res_list = []
    target_ip = addrs_list[tar_index]
    target_port = int(ports_list[tar_index])
    tar_addr = (target_ip, target_port)

    # send the layer_embedding_norm_bytes to the server
    layer_embedding_norm_bytes = socket_comm_module.pack_tensor(tensor=layer_embedding_norm)
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

def cat_res(results:list) -> list:
    """
    this function is used to cat the results from the multi-threading
    """
    q_per_token_all_heads_list = []
    k_per_token_all_heads_list = []
    v_per_token_all_heads_list = []

    for res in results:
        if res and len(res) == 3:  # 确保每个结果包含q/k/v三个元素
            q_per_token_all_heads_list.append(res[0])
            k_per_token_all_heads_list.append(res[1])
            v_per_token_all_heads_list.append(res[2])

    # cat the pieces together
    q_per_token_all_heads = concat_tensors(tensor_list=q_per_token_all_heads_list, dim=2)
    k_per_token_all_heads = concat_tensors(tensor_list=k_per_token_all_heads_list, dim=2)
    v_per_token_all_heads = concat_tensors(tensor_list=v_per_token_all_heads_list, dim=2)
    
    return q_per_token_all_heads, k_per_token_all_heads, v_per_token_all_heads

def inference_server(model, tokenizer, config, server, input_text, allocation_list, user_config):
    """
    tips: config is the config file of the model, and the user_config is the config file of the topo.
    :param model: model
    :param tokenizer: tokenizer
    :param config: model config dict
    :param server: server object
    :param input_text: input text, str
    :param allocation_list: ratios of each node
    :param user_config: user config dict, inclding ip' addrs, ports, and so on
    :return: next text predicted by LLM
    """

    # dtype is here
    dtype = torch.bfloat16

    token_embeddings_unnormalized, tokens_length = input_embedding(input_text=input_text, tokenizer=tokenizer, config=config, model=model, dtype=dtype)
    freqs_cis = get_freqs_cis(config, tokens_length)

    # how to get addrs_list?
    addrs_list = user_config["network_config"]["addrs_list"]
    ports_list = user_config["network_config"]["ports_list"]

    final_embedding = token_embeddings_unnormalized
    for layer in range(config["n_layers"]):

        print(f"layer:{layer}")

        qkv_attention_store = []
        layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"], config)

        # load model weights
        q_layer_matrix = model[f"layers.{layer}.attention.wq.weight"]
        q_layer_matrix = q_layer_matrix.view(config["n_heads"], q_layer_matrix.shape[0] // config["n_heads"], config["dim"])
        k_layer_matrix = model[f"layers.{layer}.attention.wk.weight"]
        k_layer_matrix = k_layer_matrix.view(config["n_kv_heads"], k_layer_matrix.shape[0] // config["n_kv_heads"], config["dim"])
        v_layer_matrix = model[f"layers.{layer}.attention.wv.weight"]
        v_layer_matrix = v_layer_matrix.view(config["n_kv_heads"], v_layer_matrix.shape[0] // config["n_kv_heads"], config["dim"])

        # split qkv matrix, x_chunks' type is tuple
        q_chunks = split_matrix(matrix=q_layer_matrix, ratio_list=allocation_list, dim=1)
        k_chunks = split_matrix(matrix=k_layer_matrix, ratio_list=allocation_list, dim=1)
        v_chunks = split_matrix(matrix=v_layer_matrix, ratio_list=allocation_list, dim=1)

        # multi-threading to distribute the qkv matrix
        results = [None] * len(addrs_list)
        threads = []
        for i in range(len(allocation_list)):
            thread = threading.Thread(
                target=lambda idx, r: r.__setitem__(idx, QKV_distribution(
                    addrs_list,
                    ports_list,
                    idx,
                    server,
                    q_chunks,
                    k_chunks,
                    v_chunks,
                    layer_embedding_norm
                )),
                args=(i, results)
            )
            threads.append(thread)
            thread.start()
        
        # wait for all threads to finish
        for thread in threads:
            thread.join()

        # cat the multi-nodes results
        q_per_token_all_heads, k_per_token_all_heads, v_per_token_all_heads = cat_res(results=results)

        # multi-heads attention process
        for head in range(config["n_heads"]):
            q_per_token = q_per_token_all_heads[head]
            k_per_token = k_per_token_all_heads[head//4]
            v_per_token = v_per_token_all_heads[head//4]
            v_per_token = v_per_token.to(dtype)

            q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape).to(dtype)

            k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape).to(dtype)

            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
            qk_per_token = qk_per_token.to(dtype)

            mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(dtype)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)
        
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        w_layer_matrix = model[f"layers.{layer}.attention.wo.weight"]

        # Want to add a distribution here?
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer_matrix.T)

        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"], config)
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit+output_after_feedforward
    
    # final_norm
    final_embedding = rms_norm(final_embedding, model["norm.weight"], config)

    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)

    next_token = torch.argmax(logits, dim=-1)

    next_text = tokenizer.decode([next_token.item()])

    return next_text