"""
api of communication modules using SOCKET
"""
import socket
import threading
import struct
import struct
import pickle
import io
import torch

def send_message(sock: socket.socket, data: bytes):
    """
    Send a message with a length prefix
    """
    length = len(data)
    sock.sendall(struct.pack('!I', length) + data)

def receive_message(sock: socket.socket):
    """
    Receive message with length prefix
    """
    try:
        # 接收4字节的长度信息
        length_data = _recv_all(sock, 4)
        if not length_data:
            return None
        
        # 解析长度
        length = struct.unpack('!I', length_data)[0]
        
        # 接收指定长度的数据
        return _recv_all(sock, length)
    except (ConnectionResetError, BrokenPipeError):
        return None
    
def _recv_all(sock: socket.socket, n: int):
    """
    Guaranteed to receive data of the specified length
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def read_network_config(network_config: dict):
    addrs_list = network_config["addrs_list"]
    ports_list = network_config["ports_list"]
    return addrs_list, ports_list

def pack_tensor(tensor, computation_time = None, message_type = "DATA"):
    """
    Pack a tensor and time info into bytes.
    Args:
        tensor: The tensor to be packed.
        computation_time: The time taken for computation (optional).
        message_type: The type of message (optional).
    Returns:
        tensor_bytes: The packed data as bytes.
    """
    # init the dict
    data_package = {
        "tensor": tensor,
        "message_type": message_type
    }

    if computation_time is not None:
        data_package["computation_time"] = computation_time
        data_package["message_type"] = "TIMING"
    data_package_bytes = pickle.dumps(data_package)
    return data_package_bytes

def unpack_tensor(data_package_bytes):
    """
    Args:
        packed_data: bytes data

    Returns:
        tensor: The unpacked tensor.
        computation_time: The computation time (if present).
        message_type: The type of message (if present).
    """
    # Unpack the data
    data_package = pickle.loads(data_package_bytes)

    # get tensor
    tensor_data = data_package.get("tensor")

    message_type = data_package.get("message_type", "DATA")

    computation_time = data_package.get("computation_time", None)

    return tensor_data, message_type, computation_time