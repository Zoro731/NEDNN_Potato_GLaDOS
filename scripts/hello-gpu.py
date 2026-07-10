print('Hello gpu!')

import torch

hascuda = torch.cuda.is_available()
devices = torch.cuda.device_count()

print(f"is cuda available? {hascuda}")

if hascuda:
    print(f"how many GPUs did we get? {devices}")
    print('which GPUs did we get?')
    for device in range(devices):
        print(f"device {device}: {torch.cuda.get_device_name(device)}")
