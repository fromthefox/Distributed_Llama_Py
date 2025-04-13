"""
functions of llama-3
"""
import torch
import socket_server
import socket_comm_module
import threading
import time
import numpy as np


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
    computation_timeinfo_list = []
    QKV_start = time.perf_counter()
    translation_timinfo = None
    # send the layer_embedding_norm_bytes to the server
    layer_embedding_norm_bytes = socket_comm_module.pack_tensor(tensor=layer_embedding_norm)
    response_embedding = server.send_data(tar_addr, layer_embedding_norm_bytes)
    if response_embedding == b"Received": # means the client received the embedding res
        q_chunks_bytes = socket_comm_module.pack_tensor(tensor=q_chunks[tar_index])
        response_q_per_token_all_heads_piece_bytes = server.send_data(tar_addr, q_chunks_bytes)
        response_q_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_q_per_token_all_heads_piece_bytes)
        if response_q_per_token_all_heads_piece[1] == "TIMING":
            computation_time = response_q_per_token_all_heads_piece[2]
            computation_timeinfo_list.append(computation_time)
        QKV_res_list.append(response_q_per_token_all_heads_piece[0])
        
        k_chunks_bytes = socket_comm_module.pack_tensor(tensor=k_chunks[tar_index])
        response_k_per_token_all_heads_piece_bytes = server.send_data(tar_addr, k_chunks_bytes)
        response_k_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_k_per_token_all_heads_piece_bytes)
        if response_k_per_token_all_heads_piece[1] == "TIMING":
            computation_time = response_k_per_token_all_heads_piece[2]
            computation_timeinfo_list.append(computation_time)
        QKV_res_list.append(response_k_per_token_all_heads_piece[0])
        
        v_chunks_bytes = socket_comm_module.pack_tensor(tensor=v_chunks[tar_index])
        response_v_per_token_all_heads_piece_bytes = server.send_data(tar_addr, v_chunks_bytes)
        response_v_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_v_per_token_all_heads_piece_bytes)
        if response_v_per_token_all_heads_piece[1] == "TIMING":
            computation_time = response_v_per_token_all_heads_piece[2]
            computation_timeinfo_list.append(computation_time)
        QKV_res_list.append(response_v_per_token_all_heads_piece[0])

        computation_timeinfo = sum(computation_timeinfo_list)

        QKV_end = time.perf_counter()
        QKV_sum_time = QKV_end - QKV_start
        translation_timeinfo = QKV_sum_time - computation_timeinfo
    return QKV_res_list, computation_timeinfo, translation_timeinfo

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
    print(addrs_list)
    print(ports_list)

    final_embedding = token_embeddings_unnormalized
    computation_timeinfo_for_all_nodes = []
    translation_timeinfo_for_all_nodes = []
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
        computation_timeinfo = [None] * len(addrs_list)
        translation_timeinfo = [None] * len(addrs_list)
        threads = []
        for i in range(len(allocation_list)):
            thread = threading.Thread(
                target=lambda idx, r, t, x: (
                    r.__setitem__(idx, result[0]),
                    t.__setitem__(idx, result[1]),
                    x.__setitem__(idx, result[2])
                ) if (result := QKV_distribution(
                    addrs_list,
                    ports_list,
                    idx,
                    server,
                    q_chunks,
                    k_chunks,
                    v_chunks,
                    layer_embedding_norm
                )) else None,
                args=(i, results, computation_timeinfo, translation_timeinfo)
            )
            threads.append(thread)
            thread.start()
        
        # wait for all threads to finish
        for thread in threads:
            thread.join()

        computation_timeinfo_for_all_nodes.append(computation_timeinfo)
        translation_timeinfo_for_all_nodes.append(translation_timeinfo)

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
    computation_time_list = []
    translation_time_list = []
    for i in range(len(allocation_list)):
        mid_computation_time = 0
        mid_translation_time = 0
        for j in range(len(computation_timeinfo_for_all_nodes)):
            mid_computation_time += computation_timeinfo_for_all_nodes[j][i]
            mid_translation_time += translation_timeinfo_for_all_nodes[j][i]
        computation_time_list.append(mid_computation_time)
        translation_time_list.append(mid_translation_time)

    print(f"computation_time_list:{computation_time_list}")
    print(f"translation_time_list:{translation_time_list}")
    return next_text, computation_time_list, translation_time_list

def proportinal_allocation_dis(scores_list:list, model_unsplitted_dim:int) -> list:
    """
    this func is used to allocate the conculation tast of the inference according to the scores_list.
    :param scores_list: the list of the scores, which is used to allocate the calculation task.
    :param model_unsplitted_dim: the total number wait for splitting.
    :return: the list of the allocation result.
    """
    # if not scores_list:
    #     raise ValueError("scores_list cannot be empty")
    
    if model_unsplitted_dim < 0:
        raise ValueError("model_unsplitted_dim must > 0")
    
    scores = [float(s) for s in scores_list]

    total_weight = sum(scores)

    # Calculation of the theoretical assigned value
    exact_allocations = [model_unsplitted_dim * (s / total_weight) for s in scores]

    integer_parts = [int(a) for a in exact_allocations]
    fractional_parts = [a - int(a) for a in exact_allocations]

    # Calculate the remaining number to be allocated
    total_allocated = sum(integer_parts)
    remaining = model_unsplitted_dim - total_allocated

    if remaining < 0:
        raise RuntimeError("Algorithm error: allocation value exceeds total")
    
    if remaining > 0:
        # Sort indices in descending order by fractional part
        # 从两个维度的降序排列：1. 小数部分从大到小；2. 得分从大到小
        sorted_indices = sorted(
            range(len(fractional_parts)),
            key=lambda i: (-fractional_parts[i], -scores[i])
        )
        
        # Allocate the remaining quantity
        for i in sorted_indices[:remaining]:
            integer_parts[i] += 1
    
    return integer_parts

def cal_new_dynamic_ratio(computation_time_list:list, translation_time_list:list) -> float:
    """
    this func is used to compute the new dynamic ratio based on the computation time and the communication time.
    :param computation_time_list: the computation time list from each node.
    :param comm_time_list: the communication time list from each node.
    :return: the new dynamic ratio for the nodes.
    """
    computation_times = np.array(computation_time_list)
    translation_times = np.array(translation_time_list)
    
    computation_sum_time = sum(computation_times)
    translation_sum_time = sum(translation_times)
    
    total_time = computation_sum_time + translation_sum_time
    computation_ratio = (computation_sum_time / total_time)
    translation_ratio = (translation_sum_time / total_time)

    ratio = max(computation_ratio, translation_ratio)
    ratio_final = min(max(ratio, 0.1), 0.9)
    
    
    return 1-ratio_final

def dynamic_weights_dis(dynamic_weights:np.ndarray, base_weights:np.ndarray, dynamic_ratio = 0.05) -> np.ndarray:
    """
    this func is used to compute the dynamic weights based on the dynamic weights and the base weights.
    :param dynamic_weights: the dynamic weights from the nodes.
    :param base_weights: the base weights for the nodes.
    :return: the dynamic weights for the nodes.
    """
    dynamic_weights = np.array(dynamic_weights)
    base_weights = np.array(base_weights)   
    final_weights = (1-dynamic_ratio) * base_weights + dynamic_ratio * dynamic_weights
    return final_weights / final_weights.sum()

def robust_normalize_dis(arr):
    q10 = np.percentile(arr, 10)
    q90 = np.percentile(arr, 90)
    return (arr - q10) / (q90 - q10 + 1e-8)

def total_score_dis(nodes_info_dict:dict, dynamic_weights:np.ndarray)->list:
    """
    func: compute the total score of the nodes based on 3 dimensions
    input: nodes_info_dict, including arithmetic, bandwidth and memory
    output: the list including the score of each node
    """
    # 1. Get the necessary information
    arithmetic_list, bandwidth_list, memory_list = nodes_info_dict["arithmetic"], nodes_info_dict["bandwidth"], nodes_info_dict["memory"]
    
    # mem_mask = memory_filter(memory, task_demand["memory"])
    # Memory Hard Filtering


    norm_arith = robust_normalize_dis(arithmetic_list)
    norm_bw = robust_normalize_dis(bandwidth_list)
    norm_mem = robust_normalize_dis(memory_list)

    weights = np.array(dynamic_weights)
    weights = weights / (weights.sum() + 1e-8)

    hybrid_scores = []
    for a, b, m in zip(norm_arith, norm_bw, norm_mem):
        # Arithmetic weighted guarantee basis values
        base_score = np.dot([a, b, m], weights)
        
        # Harmonize the advantages of both
        hybrid = base_score
        hybrid_scores.append(hybrid)
    
    # Softmax normalization
    hybrid_scores = np.array(hybrid_scores)
    exp_scores = np.exp(hybrid_scores - np.max(hybrid_scores))  # 减去最大值避免数值溢出
    final_scores = exp_scores / (exp_scores.sum() + 1e-8)
    return final_scores

def cal_new_base_weights(computation_time_list:list, translation_time_list:list) -> np.ndarray:
    """
    this func is used to compute the new base weights based on the computation time and the communication time.
    :param computation_time_list: the computation time list from each node.
    :param comm_time_list: the communication time list from each node.
    :return: the new base weights for the nodes.
    """
    computation_times = np.array(computation_time_list)
    translation_times = np.array(translation_time_list)
    
    computation_sum_time = sum(computation_times)
    translation_sum_time = sum(translation_times)
    
    total_time = computation_sum_time + translation_sum_time
    computation_ratio = (computation_sum_time / total_time)
    translation_ratio = (translation_sum_time / total_time)
    base_weights_list = [computation_ratio, translation_ratio, 0]
    new_base_weights = np.array(base_weights_list).flatten()

    return new_base_weights

# nodes_info_dict = {'arithmetic': [631, 631, 2301, 2301], 'memory': [8, 8, 350, 350], 'bandwidth': [160, 160, 80, 80]}
# computation_time_list = [0.8460162929996926, 0.8853536270001321, 0.5917502362281084, 0.5708553295116872]
# translation_time_list = [175.27169220706753, 236.51146707308698, 365.3462711638422, 357.4909621703555]
# dynamic_part = [0.03562304,0.93104362,0.03333333]
# new_base_weights = cal_new_base_weights(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
# new_dynamic_ratio = cal_new_dynamic_ratio(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
# dynamic_weights_array = dynamic_weights_dis(dynamic_weights=dynamic_part, base_weights=new_base_weights, dynamic_ratio=new_dynamic_ratio)
# scores_list = total_score_dis(nodes_info_dict, dynamic_weights_array)
# allocation_list = proportinal_allocation_dis(scores_list, 128)
# print(f"new_dynamic_ratio: {new_dynamic_ratio}")
# print(f"Dynamic Part: {dynamic_part}")
# print(f"Dynamic Weights: {dynamic_weights_array}")
# print(f"Final Scores: {scores_list}")
# print(f"Allocation List: {allocation_list}")