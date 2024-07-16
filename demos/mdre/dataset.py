import torch
import numpy as np
from torch.utils.data import Dataset


class DistDataset(Dataset):
    def __init__(self, p, q, m=None, num_samples=100000, alphas=[]):
        self.p = p
        self.q = q
        self.m = m
    
        print('Sampling p')
        self.p_samples = p.sample((num_samples,))
        print('Sampling q')
        self.q_samples = q.sample((num_samples,))
        print(m)

        # Linear interpolation between p and q using given alphas, if m not defined
        if m is None:
            print('Linear mixing for m samples')
#             alphas = torch.from_numpy(np.tile(torch.Tensor([0.0,6.103515625e-05,0.0078125,0.13348388671875,1.0]), (num_samples // 5,)))
#             alphas = torch.from_numpy(np.tile(torch.Tensor([0., 0.5, 0.75, 0.75, 1.0]), (num_samples // 5,)))
            alphas = torch.from_numpy(np.tile(torch.Tensor([0.5]), (num_samples,)))
            print(alphas.shape)
            
#             alphas = torch.tile(torch.Tensor([0., 0.5, 0.75, 0.75, 1.0]), (num_samples // 5,)).unsqueeze(1)
            self.m_samples = torch.sqrt(1-alphas**2)*self.p_samples + alphas*self.q_samples
            
        elif isinstance(m,list):
            self.m_samples = torch.cat([dist.sample([num_samples//len(m)]) for dist in self.m])
        else:
            print('Sampling m')
            self.m_samples = m.sample((num_samples,))
        print(self.p_samples.shape)
        print(self.q_samples.shape)
        print(self.m_samples.shape)

    def __getitem__(self, idx):
        return self.p_samples[idx], self.q_samples[idx], self.m_samples[idx]
    
    def __len__(self):
        return len(self.p_samples)

    def get_p_samples(self):
        return self.p_samples
    
    def get_q_samples(self):
        return self.q_samples
    
    def get_m_samples(self):
        return self.m_samples
