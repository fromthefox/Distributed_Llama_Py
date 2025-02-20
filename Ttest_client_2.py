import socket
import threading
from socket_comm_module import send_message, receive_message, unpack_tensor

class TCPClient:
    """客户端实现"""
    def __init__(self, host: str = '192.168.1.104', port: int = 44444):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("192.168.1.104", 33333))
        self.sock.connect((host, port))
        self.running = True

    def start_handling(self):
        """启动请求处理循环"""
        while self.running:
            try:
                data = receive_message(self.sock)
                if not data:
                    break
                
                data_tensor = unpack_tensor(data)
                if len(data_tensor.shape) == 2:
                    input_embedding = data_tensor
                    response = b'Received'
                else:
                    matrix = data_tensor
                    matrix_res = 
                    if input_embedding == None:
                        raise ValueError("input_embedding is None!!")
                    else:
                        for i in range(matrix.shape[0]):
                            matrix_head = matrix[i]

                
                send_message(self.sock, response)
            except (ConnectionResetError, BrokenPipeError):
                break

        self.sock.close()

if __name__ == "__main__":
    clients = []
    # for _ in range(3):  # 模拟多个客户端
    #     client = TCPClient()
    #     threading.Thread(target=client.start_handling).start()
    #     clients.append(client)
    client = TCPClient()
    client.start_handling()