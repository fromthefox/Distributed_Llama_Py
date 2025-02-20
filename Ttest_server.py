import socket_client
import socket_server
import torch
import threading
import time
import model_inference_module
import socket_comm_module

addrs_list = [
    ('192.168.1.104', 55555),
    ('192.168.1.104', 33333)
]
ratios_list = [1, 2]
input_embedding = torch.randn(18, 3)
tensor_q = torch.randn(1, 6, 3)
print(tensor_q)

server = socket_server.TCPServer(port=44444)
server_thread = threading.Thread(target=server.start)
server_thread.start()
time.sleep(20)

q_chunks = model_inference_module.split_matrix(matrix=tensor_q, ratios_list=ratios_list, dim=1)
q_per_token_all_heads_list = []

for i in range(len(ratios_list)):
    target_addr = addrs_list[i]
    input_embedding_bytes = socket_comm_module.pack_tensor(input_embedding)
    response_embedding = server.send_data(target_addr, input_embedding_bytes)
    print(response_embedding)
    if response_embedding == b"Received":
        q_chunks_bytes = socket_comm_module.pack_tensor(q_chunks[i])
        response_q_per_token_all_heads_piece_bytes = server.send_data(target_addr, q_chunks_bytes)
        response_q_per_token_all_heads_piece = socket_comm_module.unpack_tensor(response_q_per_token_all_heads_piece_bytes)
        print(f"response_q_per_token_all_heads_piece from {target_addr}")
        print(response_q_per_token_all_heads_piece)
        q_per_token_all_heads_list.append(response_q_per_token_all_heads_piece)

q_per_token_all_heads = model_inference_module.concat_tensors(tensor_list=q_per_token_all_heads_list, dim=2)

print(q_per_token_all_heads)
print("--------------------------------------")
real_res = torch.matmul(input_embedding, tensor_q[0].T)
print(real_res)