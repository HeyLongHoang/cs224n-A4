# from utils import pad_sents

# sents = [
#     ['Hello', 'how', 'are', 'you'],
#     ['Hola']
# ]
# unk_token = '<UNK>'

# pad_sts = pad_sents(sents, unk_token)
# print(pad_sts)

import torch

y = torch.zeros((10,3,20))
x = torch.zeros((10,20))
# print(x.unsqueeze(-1).shape)
print(torch.bmm(y, x.unsqueeze(-1)).squeeze(-1).shape)
