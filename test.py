import torch


def euclidean_dist(x, y):
    """Compute squared Euclidean distances between rows of two tensors."""
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be two-dimensional tensors.")
    if x.size(1) != y.size(1):
        raise ValueError("x and y must have the same feature dimension.")

    return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2)


if __name__ == "__main__":
    x = torch.tensor([[1, 1, 1]])
    y = torch.tensor([[2, 2, 2]])
    print(euclidean_dist(x, y))
