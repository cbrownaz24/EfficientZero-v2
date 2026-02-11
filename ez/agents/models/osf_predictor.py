"""
Spectral Dynamics Network with Observation Spectral Filtering (OSF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OSFPredictor(nn.Module):
    def __init__(self, T, h, m, D_in, D_out):
        super().__init__()
        self.T = T
        self.h = h
        self.m = m
        self.D_in = D_in
        self.D_out = D_out
        
        # Hankel matrix Z_{ij} = 2 / ((i+j)^3 - (i+j)), i,j >= 1
        sigma, phi = self._compute_hankel_eigenpairs(T, h)
        self.register_buffer('sigma_qrt', sigma.pow(0.25))  # (h,)
        self.register_buffer('phi', phi)                      # (h, T) or (h, T-1)
        
        self.J = nn.Parameter(torch.zeros(m, D_out, D_in))
        self.M = nn.Parameter(torch.zeros(h, D_out, D_in))
        self.P = nn.Parameter(torch.zeros(m, D_out, D_out))
        self.N = nn.Parameter(torch.zeros(h, D_out, D_out))
        
        self._init_parameters()
    
    def _init_parameters(self):
        for param in [self.J, self.M, self.P, self.N]:
            nn.init.normal_(param, std=1e-3)
    
    def _compute_hankel_eigenpairs(self, size, k):
       if size < 1:
            return torch.ones(k), torch.ones(k, max(size, 1))
        
        idx = torch.arange(1, size + 1, dtype=torch.float64)
        s = idx.unsqueeze(0) + idx.unsqueeze(1)
        Z = 2.0 / (s ** 3 - s)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(Z)
        
        actual_k = min(k, size)
        topk_idx = torch.argsort(eigenvalues, descending=True)[:actual_k]
        sigma = eigenvalues[topk_idx].float().clamp(min=1e-12)
        phi = eigenvectors[:, topk_idx].T.float()  # (actual_k, size)
        
        if actual_k < k:
            pad_sigma = torch.full((k - actual_k,), 1e-12)
            pad_phi = torch.zeros(k - actual_k, size)
            sigma = torch.cat([sigma, pad_sigma])
            phi = torch.cat([phi, pad_phi], dim=0)
        
        return sigma, phi
    
    def forward(self, u_seq, y_seq):
        # u_seq: (B, T, D_in)
        # y_seq: (B, T, D_out)
        # y_hat: (B, D_out)

        B = u_seq.shape[0]
        T = u_seq.shape[1]
        device = u_seq.device
        dtype = u_seq.dtype
        
        y_hat = torch.zeros(B, self.D_out, device=device, dtype=dtype)
        
        # sum_{j=1}^{m-1} J_j u_{T-1-j}
        ar_u_count = min(self.m - 1, T - 1)
        if ar_u_count > 0:
            # j=1 -> u_{T-2}, j=2 -> u_{T-3}, ...
            idx = torch.arange(T - 2, T - 2 - ar_u_count, -1,
                               device=device).clamp(min=0)
            u_recent = u_seq[:, idx]  # (B, ar_u_count, D_in)
            
            y_hat += torch.einsum('jod, bjd -> bo',
                                  self.J[:ar_u_count], u_recent)
        
        # sum_{i=1}^{h} sigma_i^{1/4} M_i <phi_i, u_{T-2:0}>
        if self.h > 0 and T >= 2:
            # u window: [u_{T-2}, u_{T-3}, ..., u_0] length T-1
            u_window = u_seq[:, :T - 1].flip(1)  # (B, T-1, D_in)
            
            phi_u = self.phi[:, :T - 1]  # (h, T-1)
            
            # (h, T-1) x (B, T-1, D_in) -> (B, h, D_in)
            spectral_u = torch.einsum('hl, bld -> bhd', phi_u, u_window)
            
            # (h,) * (h, D_out, D_in) x (B, h, D_in) -> (B, D_out)
            y_hat += torch.einsum('h, hod, bhd -> bo',
                                  self.sigma_qrt, self.M, spectral_u)
        
        # sum_{j=1}^{m} P_j y_{T-1-j}
        ar_y_count = min(self.m, T - 1)
        if ar_y_count > 0:
            idx = torch.arange(T - 2, T - 2 - ar_y_count, -1,
                               device=device).clamp(min=0)
            y_recent = y_seq[:, idx]  # (B, ar_y_count, D_out)
            
            y_hat += torch.einsum('joe, bje -> bo',
                                  self.P[:ar_y_count], y_recent)
        
        # sum_{i=1}^{h} sigma_i^{1/4} N_i <phi_i, y_{T-1:0}>
        if self.h > 0 and T >= 1:
            # y window: [y_{T-1}, y_{T-2}, ..., y_0] length T
            y_window = y_seq.flip(1)  # (B, T, D_out)
            
            phi_y = self.phi[:, :T]  # (h, T)
            
            # (h, T) x (B, T, D_out) -> (B, h, D_out)
            spectral_y = torch.einsum('hl, ble -> bhe', phi_y, y_window)
            
            # (B, D_out)
            y_hat += torch.einsum('h, hoe, bhe -> bo',
                                  self.sigma_qrt, self.N, spectral_y)
        
        return y_hat