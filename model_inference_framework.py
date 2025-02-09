"""
framework of distributed-llama-3 inference
"""
import model_inference_module
import init
import socket_server
import socket_client
import matrix_split
import socket_comm_module
import threading
import time
import torch

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

    final_embedding = token_embeddings_unnormalized
    for layer in range(config.n_layers):
        qkv_attention_store = []
        layer_embedding_norm = model_inference_module.rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
        q_layer = model[f"layers.{layer}.attention.wq.weight"]
        q_layer = q_layer.view(config.n_heads, q_layer.shape[0] // config.n_heads, config.dim)
        k_layer = model[f"layers.{layer}.attention.wk.weight"]
        k_layer = k_layer.view(config.n_kv_heads, k_layer.shape[0] // config.n_kv_heads, config.dim)
        v_layer = model[f"layers.{layer}.attention.wv.weight"]
        v_layer = v_layer.view(config.n_kv_heads, v_layer.shape[0] // config.n_kv_heads, config.dim)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        # 这里直接把qkv权重矩阵发送给其他node, 以及layer_embedding_norm

        # 后面的head的循环操作在各个node执行

        # ---- 在这里直接把Embedding结果和q/k/v_layer广播给node, 然后node全部计算完直接返回给root ----

        # ---- ----
        for head in range(config.n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head//4]
            v_layer_head = v_layer[head//4]

            # ---- 分发 ----

            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

            # q_chunks = matrix_split.split_matrix(q_layer_head, ratios_list, 0)
            # k_chunks = matrix_split.split_matrix(k_layer_head, ratios_list, 0)
            # v_chunks = matrix_split.split_matrix(v_layer_head, ratios_list, 0)


            # ---- 收集 ----

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
