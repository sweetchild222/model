import torch
print(f"get device name : {torch.cuda.get_device_name(0)}")
print(f"is available : {torch.cuda.is_available()}")
print(f"torch version : {torch.__version__}")
