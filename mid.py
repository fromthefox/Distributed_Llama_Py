import torch
from model_inference_module import split_matrix

def test_split_matrix():
    # 创建一个示例 3D 张量
    # 形状为 [4, 6, 8]，分别表示 n_heads, head_dim, embed_dim
    tensor = torch.arange(192).reshape(4, 6, 8)
    print("原始张量形状:", tensor.shape)
    print("原始张量第一个头:")
    print(tensor[0])
    print("\n原始张量第二个头:")
    print(tensor[1])
    
    # 测试沿维度 1 (head_dim) 切分
    # 将第二个维度按照 [2, 4] 的比例切分
    chunks = split_matrix(tensor, ratio_list=[2, 4], dim=1)
    
    print("\n切分后得到", len(chunks), "个张量")
    print("第一个切片形状:", chunks[0].shape)
    print("第二个切片形状:", chunks[1].shape)
    
    print("\n第一个切片内容 (第一个头):")
    print(chunks[0][0])
    print("\n第二个切片内容 (第一个头):")
    print(chunks[1][0])
    
    # 验证切分后拼接是否与原始张量一致
    reconstructed = torch.cat(chunks, dim=1)
    print("\n重建后与原始张量是否相同:", torch.all(reconstructed == tensor).item())
    
    # 测试不同维度的切分
    dim0_chunks = split_matrix(tensor, ratio_list=[1, 3], dim=0)
    print("\n沿维度0切分形状:", [chunk.shape for chunk in dim0_chunks])
    
    dim2_chunks = split_matrix(tensor, ratio_list=[3, 5], dim=2)
    print("\n沿维度2切分形状:", [chunk.shape for chunk in dim2_chunks])

if __name__ == "__main__":
    test_split_matrix()