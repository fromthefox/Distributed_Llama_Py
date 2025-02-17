"""
matrix_split.py
split the matrix into uneven pieces
"""
import torch

def split_matrix(matrix, ratios_list, dim):
    """
    Split the matrix along specified dimension and return the chunks.
    matrix: raw matrix, tensor, n dims
    ratios_list: the split ratio of matrix, list
    dim: the dimension to split (0-based index)
    """
    pieces_num = sum(ratios_list)
    shape_dim = matrix.shape[dim]
    
    if shape_dim % pieces_num != 0:
        raise ValueError(f"Dimension {dim} ({shape_dim}) not divisible by {pieces_num}")
    
    piece_size = shape_dim // pieces_num
    split_tuple = tuple(i * piece_size for i in ratios_list)
    
    # Validate split_tuple
    if sum(split_tuple) != shape_dim:
        raise ValueError(f"split_tuple {split_tuple} sum does not match shape dimension {shape_dim}")
    
    chunks = torch.split(tensor=matrix, split_size_or_sections=split_tuple, dim=dim)
    
    # chunks is a tuple type
    return chunks
