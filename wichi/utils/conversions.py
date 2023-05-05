import torch


def value_mat_to_torch_tensor(wichi_mat, dtype=torch.double):
    m = len(wichi_mat)
    n = len(wichi_mat[0])

    tensor = torch.empty((m, n), dtype=dtype)
    for row in range(m):
        for col in range(n):
            tensor[row, col] = wichi_mat[row][col].data

    return tensor
