from .objective import *
torch.set_num_threads(1)
rdBase.DisableLog('rdApp.error')
dev = torch.device('cuda')
devices = [1]
