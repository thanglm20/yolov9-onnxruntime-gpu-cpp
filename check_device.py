import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
for i in range(0,torch.cuda.device_count()):
    print(torch.cuda.device(i))
    print(torch.cuda.get_device_name(i))