import torch

def get_device(rank: int) -> torch.device:
    """CPU or CUDA"""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

if __name__ == '__main__':
    device = get_device(0)
    print(device)