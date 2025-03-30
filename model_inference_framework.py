"""
this file is used to finish the inference task
"""

import model_inference_module
from model_inference_module import QKV_distribution
import threading
import torch

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
    token_embeddings_unnormalized, tokens_length = model_inference_module.input_embedding(input_text=input_text, tokenizer=tokenizer, config=config, model=model)
    freqs_cis = model_inference_module.get_freqs_cis(config, tokens_length)

    # how to get addrs_list?
    addrs_list = user_config["addrs"]

    final_embedding = token_embeddings_unnormalized
    for layer in range(config.n_layers):
        qkv_attention_store = []
        layer_embedding_norm = model_inference_module.rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])

        # load model weights
        q_layer_matrix = model[f"layers.{layer}.attention.wq.weight"]
        q_layer_matrix = q_layer_matrix.view(config.n_heads, q_layer_matrix.shape[0] // config.n_heads, config.dim)
        k_layer_matrix = model[f"layers.{layer}.attention.wk.weight"]
        k_layer_matrix = k_layer_matrix.view(config.n_kv_heads, k_layer_matrix.shape[0] // config.n_kv_heads, config.dim)
        v_layer_matrix = model[f"layers.{layer}.attention.wv.weight"]
        v_layer_matrix = v_layer_matrix.view(config.n_kv_heads, v_layer_matrix.shape[0] // config.n_kv_heads, config.dim)

        # split qkv matrix, x_chunks' type is tuple
        q_chunks = model_inference_module.split_matrix(matrix=q_layer_matrix, ratio_list=allocation_list, dim=1)
        k_chunks = model_inference_module.split_matrix(matrix=k_layer_matrix, ratio_list=allocation_list, dim=1)
        v_chunks = model_inference_module.split_matrix(matrix=v_layer_matrix, ratio_list=allocation_list, dim=1)

        # multi-threading to distribute the qkv matrix
        results = [None] * len(addrs_list)
        threads = []
        for i in range(len(allocation_list)):
            thread = threading.Thread(
                target=lambda idx, r: r.__setitem__(idx, QKV_distribution(
                    allocation_list,
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
        
        # wait for all threads to finish
        for thread in threads:
            thread.join()

        # cat the multi-nodes results
        q_per_token_all_heads, k_per_token_all_heads, v_per_token_all_heads = model_inference_module.cat_res(results=results)

        # multi-heads attention process
        for head in range(config.n_heads):
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
        w_layer_matrix = model[f"layers.{layer}.attention.wo.weight"]

        # Want to add a distribution here?
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer_matrix.T)

        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = model_inference_module.rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit+output_after_feedforward
    
    # final_norm
    final_embedding = model_inference_module.rms_norm(final_embedding, model["norm.weight"])

    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)

    next_token = torch.argmax(logits, dim=-1)

    next_text = tokenizer.decode([next_token.item()])

    return next_text