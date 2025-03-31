"""
the whole framework of the project
here we accept the param from Auto-Deployment Proj.
:param accepted: list, like [25, 48, 55] for unsplitted-dim == 128
"""
import init
import socket_server
import threading
import model_inference_framework


def infenerce_main_for_server(allocation_list:list, model_path:str, tokenizer_path:str, config_path:str, user_config_path:str) -> None:
    """
    the whole framework of the project.
    :parma allocation_list: list, like [25, 48, 55] for unsplitted-dim == 128
    :parma config_path: str, the path to config file
    :return: None, just do the inference
    """

    # 1. load model, tokenizer, config
    model, tokenizer, config = init.load_model("model_path", "tokenizer_path", "config_path")
    user_config_dict = init.load_user_config(user_config_path)


    # 2. prepare the server
    
    # 2.1. init the server
    server = socket_server.TCPServer(port=9999)

    # 2.2 start listening
    server_thread = threading.Thread(target=server.start)
    server_thread.start()

    # 3. start the inference

    # 3.1. get the input text and max token
    input_text = user_config_dict["user_config"]["input_text"]
    max_token_length = user_config_dict["user_config"]["max_token_length"]

    # 3.2. start inference
    full_output = model_inference_framework.generation_loop(
        initial_input=input_text,
        max_tokens_length=max_token_length,
        model=model,
        tokenizer=tokenizer,
        config=config,
        server=server,
        allocation_list=allocation_list,
        user_config=user_config_dict
    )

    return full_output