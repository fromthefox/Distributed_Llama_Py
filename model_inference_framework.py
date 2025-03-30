"""
this file is used to finish the inference task
"""

import model_inference_module

def inference_server(model, tokenizer, config, server, input_text, allocation_list, user_config):
    """
    tips: config is the config file of the model, and the user_config is the config file of the topo.
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
        q_layer_matrix = q_layer_matrix.view(config.n_heads, q_layer.shape[0] // config.n_heads, config.dim)
        k_layer_matrix = model[f"layers.{layer}.attention.wk.weight"]
        k_layer_matrix = k_layer_matrix.view(config.n_kv_heads, k_layer.shape[0] // config.n_kv_heads, config.dim)
        v_layer_matrix = model[f"layers.{layer}.attention.wv.weight"]
        v_layer_matrix = v_layer_matrix.view(config.n_kv_heads, v_layer.shape[0] // config.n_kv_heads, config.dim)
        w_layer_matrix = model[f"layers.{layer}.attention.wo.weight"]

        # split qkv matrix
