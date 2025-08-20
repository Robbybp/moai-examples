## Running this on GPU

To see what GPU a given partition has, run:
```
sinfo -p NODENAME -o "%50N %10c %20m %30G"
```

To get info about the GPU from torch, run
```python
# n is the device number
torch.cuda.get_device_properties(n)
```

Note that the `major` and `minor` attributes specify the CUDA version
that this GPU is enabled up to. Recent torch versions may not support
sufficiently old CUDA versions, so I may need to install older torch.
I'll need to install a compatible torchvision. I can refer to
the compatibility table [here](https://pypi.org/project/torchvision/).
