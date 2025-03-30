"""
the whole framework of the project
here we accept the param from Auto-Deployment Proj.
:param accepted: list, like [25, 48, 55] for unsplitted-dim == 128
"""
import init
import socket_server
import threading

def infenerce_framework(allocation_list:list, model_path:str, tokenizer_path:str, config_path:str) -> None:
    """
    the whole framework of the project.
    :parma allocation_list: list, like [25, 48, 55] for unsplitted-dim == 128
    :parma config_path: str, the path to config file
    :return: None, just do the inference
    """

    # 1. load model, tokenizer, config
    model, tokenizer, config = init.load_file("model_path", "tokenizer_path", "config_path")

    # 2. prepare the server
    
    # 2.1. init the server
    server = socket_server.TCPServer(port=9999)

    # 2.2 start listening
    server_thread = threading.Thread(target=server.start)
    server_thread.start()

    # 3. start the inference

    # 3.1. 
