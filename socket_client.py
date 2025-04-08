import socket
import threading
from socket_comm_module import send_message, receive_message
from socket_comm_module import unpack_tensor, pack_tensor
import torch
import time

class TCPClient:
    """
    Client-side implementation
    """
    def __init__(self, host: str = '192.168.1.104', port: int = 44444):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind to a specific address and port (optional), make sure to use a different port than the server
        self.sock.bind(("192.168.1.105", 34567))

        # Connect to the server
        self.sock.connect((host, port))

        # Set the socket to non-blocking mode (optional)
        self.running = True

    def start_handling(self):
        """
        Starting the request processing loop
        """
        while self.running:
            try:
                data = receive_message(self.sock)
                if not data:
                    break
                
                data_unpack = unpack_tensor(data)
                data_tensor = data_unpack["tensor"]
                if len(data_tensor.shape) == 2:
                    """
                    here we can also pack some str with the data to differ the tensor type from input_embedding and matrix
                    PLAN A: add a flag to the tensor to indicate its type
                    PLAN B: use a dictionary to pack the tensor and the type
                    """
                    # if the tensor is 2D, it is the input embedding
                    input_embedding = data_tensor
                    print('Received')
                    response = b'Received'
                else:
                    computation_start = time.perf_counter()
                    matrix = data_tensor
                    matrix_res = torch.empty(matrix.shape[0], input_embedding.shape[0], matrix.shape[1])
                    if input_embedding == None:
                        # we must ensure that the input_embedding is not None
                        raise ValueError("input_embedding is None!!")
                    else:
                        for i in range(matrix.shape[0]):
                            matrix_res[i] = torch.matmul(input_embedding, matrix[i].T)
                    computation_end = time.perf_counter()
                    computation_time = computation_end - computation_start
                    print("---------------------\n")
                    print(f"computation time: {computation_time}")
                    print("---------------------\n")
                    response = pack_tensor(matrix_res, computation_time, "TIMING")
                
                send_message(self.sock, response)
            except (ConnectionResetError, BrokenPipeError):
                break

        self.sock.close()