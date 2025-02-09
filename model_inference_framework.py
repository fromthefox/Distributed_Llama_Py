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

def inference_server(network_config: dict) -> None:
    model, tokenizer, config = init.load_file("model_path", "tokenizer_path", "config_path")
    server = socket_server.TCPServer(port = 9999)
    server_thread = threading.Thread(target=server.start)
    server_thread.start()
    # ----
    time.sleep(10)
    # ----
