"""
framework of distributed-llama-3 inference
"""
import model_inference_module
import init
import socket_server
import socket_client
import socket_comm_module
import threading
import time
import torch

def QKV_distribution(addrs_list:list, tar_index:int, server: socket_server.TCPServer, q_chunks:tuple, k_chunks:tuple, v_chunks:tuple) -> list:
    QKV_res_list = []

    tar_addr = addrs_list[tar_index]
    layer_embedding_norm_bytes = socket_comm_module.pack_tensor(tensor=layer_embedding_norm_bytes)
    response_embedding = server.send_data(tar_addr, layer_embedding_norm_bytes)
    if response_embedding == b"Received":
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


def inference_server(user_config: dict) -> None:
    model, tokenizer, config = init.load_file("model_path", "tokenizer_path", "config_path")
    server = socket_server.TCPServer(port = 9999)
    server_thread = threading.Thread(target=server.start)
    server_thread.start()
    # ----
    time.sleep(10)
    # ----
    input_text = "This is a test demo"
    token_embeddings_unnormalized, tokens_length = model_inference_module.input_embedding(input_text=input_text, tokenizer=tokenizer, config=config, model=model)
    
    zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
    freqs = 1.0 / (config.rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(tokens_length), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    ratios_list = user_config["ratios"]
    addrs_list = user_config["addrs"]

    final_embedding = token_embeddings_unnormalized
    for layer in range(config.n_layers):
        qkv_attention_store = []
        layer_embedding_norm = model_inference_module.rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
        
        # load model weights
        q_layer = model[f"layers.{layer}.attention.wq.weight"]
        q_layer = q_layer.view(config.n_heads, q_layer.shape[0] // config.n_heads, config.dim)
        k_layer = model[f"layers.{layer}.attention.wk.weight"]
        k_layer = k_layer.view(config.n_kv_heads, k_layer.shape[0] // config.n_kv_heads, config.dim)
        v_layer = model[f"layers.{layer}.attention.wv.weight"]
        v_layer = v_layer.view(config.n_kv_heads, v_layer.shape[0] // config.n_kv_heads, config.dim)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        # layer_embedding_norm.shape -> input_length x 4096
        # q_layer.shape -> 32 x 128 x 4096
        # k_layer.shape and v_layer.shape -> 8 x 128 x 4096

        # ---- 在这里直接把Embedding结果(layer_embedding_norm)和q/k/v_layer广播给node, 然后node全部计算完直接返回给root ----
        # ---- 相当于Root 拿到的是所有的q_per_token|k_per_token|v_per_token
        # ---- 然后Root再计算后续的操作 得到需要的qkv_attention_store
        # ---- 然后考虑w是否分布式
        # 把这部分实现——Deadline@2.23

        # split qkv matrix
        q_chunks = model_inference_module.split_matrix(matrix=q_layer, ratios_list=ratios_list, dim=1)
        k_chunks = model_inference_module.split_matrix(matrix=k_layer, ratios_list=ratios_list, dim=1)
        v_chunks = model_inference_module.split_matrix(matrix=v_layer, ratios_list=ratios_list, dim=1)
        
        results = [None] * len(addrs_list)
        threads = []
        for i in range(len(ratios_list)):
            thread = threading.Thread(
                target=lambda idx, r: r.__setitem__(idx, QKV_distribution(
                    ratios_list,
                    addrs_list,
                    idx,
                    server,
                    q_chunks,
                    k_chunks,
                    v_chunks
                )),
                args=(i, results)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        q_per_token_all_heads_list = []
        k_per_token_all_heads_list = []
        v_per_token_all_heads_list = []

        for res in results:
            if res and len(res) == 3:  # 确保每个结果包含q/k/v三个元素
                q_per_token_all_heads_list.append(res[0])
                k_per_token_all_heads_list.append(res[1])
                v_per_token_all_heads_list.append(res[2])


        # cat the pieces together
        q_per_token_all_heads = model_inference_module.concat_tensors(tensor_list=q_per_token_all_heads_list, dim=2)
        k_per_token_all_heads = model_inference_module.concat_tensors(tensor_list=k_per_token_all_heads_list, dim=2)
        v_per_token_all_heads = model_inference_module.concat_tensors(tensor_list=v_per_token_all_heads_list, dim=2)

        for head in range(config.n_heads):
            # q_layer_head = q_layer[head]
            # k_layer_head = k_layer[head//4]
            # v_layer_head = v_layer[head//4]

            # q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            # k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            # v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

            q_per_token = q_per_token_all_heads[0]
            k_per_token = k_per_token_all_heads[head//4]
            v_per_token = v_per_token_all_heads[head//4]

            q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

            k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
            mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)

        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = model_inference_module.rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        # ---- 分发 ----
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        # ---- 收集 ----
        final_embedding = embedding_after_edit+output_after_feedforward
