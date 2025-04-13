"""
the whole framework of the project
here we accept the param from Auto-Deployment Proj.
:param accepted: list, like [25, 48, 55] for unsplitted-dim == 128
"""
import init
import socket_server
import threading
import model_inference_framework
import numpy as np


def infenerce_main_for_server(allocation_list:list, model_path:str, tokenizer_path:str, config_path:str, user_config_path:str, dynamic_part:np.ndarray, nodes_info_dict:dict) -> str:
    """
    the whole framework of the project.
    :parma allocation_list: list, like [25, 48, 55] for unsplitted-dim == 128
    :parma config_path: str, the path to config file
    :return: None, just do the inference
    """

    # 1.
    
    # 1.1. init the server
    server = socket_server.TCPServer(port=44444)

    # 1.2 start listening
    server_thread = threading.Thread(target=server.start)
    server_thread.start()

    # 2. load model, tokenizer, config
    model, tokenizer, config = init.load_model(model_path, tokenizer_path, config_path)
    user_config_dict = init.parse_ini_file(user_config_path)

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
        user_config=user_config_dict,
        dynamic_part=dynamic_part,
        nodes_info_dict=nodes_info_dict
    )

    return full_output

infenerce_main_for_server(
    allocation_list=[32, 32, 32, 32],
    model_path=r"C:\Users\yhbia\Desktop\边彦晖\Proj\Meta_llama\Meta-Llama-3-8B\original\consolidated.00.pth",
    tokenizer_path=r"C:\Users\yhbia\Desktop\边彦晖\Proj\Meta_llama\Meta-Llama-3-8B\original\tokenizer.model",
    config_path=r"C:\Users\yhbia\Desktop\边彦晖\Proj\Meta_llama\Meta-Llama-3-8B\original\params.json",
    user_config_path=r"C:\Users\yhbia\Desktop\学校\25.6.30-边彦晖-毕业设计\auto_deployment_summary\LLM_Auto_Deployment\Distributed_Llama_Py\user_config.ini",
    dynamic_part=np.array([0.5, 0.4, 0.1]),
    nodes_info_dict={'arithmetic': [603, 603, 2301, 2301], 'memory': [8, 8, 350, 350], 'bandwidth': [75, 75, 64, 64]}
)