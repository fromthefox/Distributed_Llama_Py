"""
this file is used to finish the inference task for worker node
"""
import socket
from socket_client import TCPClient

"""
if __name__ == "__main__":
    clients = []
    # for _ in range(3):  # 模拟多个客户端
    #     client = TCPClient()
    #     threading.Thread(target=client.start_handling).start()
    #     clients.append(client)
    client = TCPClient()
    client.start_handling()
"""


def inference_main_for_worker():
    """
    this function is used to finish the inference task for worker node
    """
    client = TCPClient()
    client.start_handling()