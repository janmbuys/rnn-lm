
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(ts, use_cuda):
  if use_cuda:
    return Variable(ts).cuda()
  else:
    return Variable(ts)

def to_numpy(dist):
  return dist.type(torch.FloatTensor).data.numpy()


def get_sentence_data_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor for source in source_list], 1)
  if use_cuda:
    data = Variable(data_ts, volatile=evaluation).cuda()
  else:
    data = Variable(data_ts, volatile=evaluation)
  return data


def get_sentence_batch(source_list, use_cuda, evaluation=False):
  data_ts = torch.cat([source.word_tensor[:-1] for source in source_list], 1)
  target_ts = torch.cat([source.word_tensor[1:] for source in source_list], 1)
  if use_cuda:
    data = Variable(data_ts, volatile=evaluation).cuda()
    target = Variable(target_ts.view(-1)).cuda()
  else:
    data = Variable(data_ts, volatile=evaluation)
    target = Variable(target_ts.view(-1))
  return data, target

