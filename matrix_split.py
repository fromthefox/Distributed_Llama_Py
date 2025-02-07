"""
matrix_split.py
split the matrix into uneven pieces
"""
import torch

def split_matrix(matrix, ratio, IS_ROW):
    """
    split the matrix and return the res.
    matrix: raw matrix, tensor, 2 dims
    ratio: the split ratio of matrix, list
    IS_ROW: 1 means row_split, 0 means col_split, bool
    """
    pieces_num = sum(ratio)
    if IS_ROW == 1:
        if matrix.shape[0] % pieces_num != 0:
            raise ValueError("column not divisible")
        piece_size = matrix.shape[0] // pieces_num
        # // means the ans is int type
        split_tuple = tuple(i * piece_size for i in ratio)
        print(split_tuple)
        chunks = torch.split(tensor=matrix, split_size_or_sections=split_tuple, dim=0)
    elif IS_ROW == 0:
        if matrix.shape[1] % pieces_num != 0:
            raise ValueError("row not divisible")
        piece_size = matrix.shape[1] // pieces_num
        split_tuple = tuple(i * piece_size for i in ratio)
        chunks = torch.split(tensor=matrix, split_size_or_sections=split_tuple, dim=1)
    else:
        raise ValueError("IS_ROW value error")
    
    # chunks is tuple type
    return chunks
    
    