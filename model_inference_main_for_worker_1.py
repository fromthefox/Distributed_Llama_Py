"""
this file is used to finish the inference task for worker node
"""
import socket
from socket_client_1 import TCPClient

def inference_main_for_worker():
    """
    this function is used to finish the inference task for worker node
    """
    host, port = "192.168.1.105", 44444
    client = TCPClient(host, port)
    client.start_handling()

inference_main_for_worker()