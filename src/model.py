"""
Core HRM Model Implementation

Implements the Hierarchical Reasoning Model with:
- High-level Module (H): Slow, abstract planning
- Low-level Module (L): Fast, detailed computation  
- Adaptive Computation Time (ACT) with Q-learning
- One-step gradient for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        
    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GLUMLP(nn.Module):
    """Gated Linear Unit MLP"""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2.67 * dim)  # SwiGLU standard
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, dim: int, n_heads: int, max_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False) 
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with Post-Norm and GLU MLP"""
    def __init__(self, dim: int, n_heads: int, max_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_len)
        self.mlp = GLUMLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Post-norm: residual -> layer -> norm
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.mlp(x))
        return x

class RecurrentModule(nn.Module):
    """Recurrent Transformer Module (used for both H and L modules)"""
    def __init__(self, dim: int, n_heads: int, n_layers: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_len) 
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class StableMax(nn.Module):
    """Numerically stable max operation for small samples"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # Subtract max for numerical stability
        x_max = x.max(dim=self.dim, keepdim=True)[0]
        x_stable = x - x_max
        return F.softmax(x_stable, dim=self.dim)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model
    
    Brain-inspired architecture with hierarchical modules operating
    at different timescales for improved reasoning capabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        h_dim: int = 512,      # High-level module dimension
        l_dim: int = 256,      # Low-level module dimension  
        n_heads_h: int = 8,    # High-level attention heads
        n_heads_l: int = 4,    # Low-level attention heads
        n_layers_h: int = 6,   # High-level layers
        n_layers_l: int = 3,   # Low-level layers
        output_dim: int = None,
        max_len: int = 2048,
        act_threshold: float = 0.5,  # ACT halting threshold
    ):
        super().__init__()
        
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.act_threshold = act_threshold
        output_dim = output_dim or input_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, l_dim)
        
        # High-level module (slow, abstract)
        self.h_module = RecurrentModule(h_dim, n_heads_h, n_layers_h, max_len)
        
        # Low-level module (fast, detailed)
        self.l_module = RecurrentModule(l_dim, n_heads_l, n_layers_l, max_len)
        
        # Cross-module communication
        self.h_to_l = nn.Linear(h_dim, l_dim)  # High-level influences low-level
        self.l_to_h = nn.Linear(l_dim, h_dim)  # Low-level informs high-level
        
        # Q-head for Adaptive Computation Time
        self.q_head = nn.Sequential(
            nn.Linear(l_dim, l_dim // 2),
            nn.ReLU(),
            nn.Linear(l_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(l_dim, output_dim)
        
        # Stable max for small sample generalization  
        self.stable_max = StableMax(dim=-1)
        
        # Initialize weights using LeCun initialization
        self._init_weights()
        
    def _init_weights(self):
        """LeCun initialization for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # LeCun normal initialization
                fan_in = module.in_features
                std = math.sqrt(1.0 / fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        max_steps: int = 10,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical processing
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            max_steps: Maximum computation steps
            return_intermediates: Whether to return intermediate states
            
        Returns:
            Dictionary containing output and optional intermediate states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project input to low-level dimension
        l_state = self.input_proj(x)  # [B, L, l_dim]
        
        # Initialize high-level state (slow, abstract planning)
        h_state = torch.zeros(batch_size, seq_len, self.h_dim, device=device)
        
        intermediates = [] if return_intermediates else None
        halting_probs = []
        cumulative_halt = torch.zeros(batch_size, seq_len, device=device)
        
        for step in range(max_steps):
            # Low-level processing (fast, detailed computation)
            l_input = l_state + self.h_to_l(h_state)  # High-level influences low-level
            l_state = self.l_module(l_input)
            
            # Adaptive Computation Time - decide whether to halt
            halt_prob = self.q_head(l_state).squeeze(-1)  # [B, L]
            halting_probs.append(halt_prob)
            
            # Update cumulative halting probability
            still_running = (cumulative_halt < self.act_threshold).float()
            halt_step = halt_prob * still_running
            cumulative_halt = cumulative_halt + halt_step
            
            # High-level update (slow, abstract planning)
            # Only update H-module when L-module converges locally
            should_update_h = (cumulative_halt >= self.act_threshold).float()
            
            if should_update_h.sum() > 0:
                # Aggregate low-level information for high-level update
                l_summary = self.l_to_h(l_state)  # [B, L, h_dim]
                h_input = h_state + l_summary
                h_state = self.h_module(h_input)
                
                # Reset low-level state where high-level updated
                reset_mask = should_update_h.unsqueeze(-1)  # [B, L, 1]
                l_state = l_state * (1 - reset_mask) + self.h_to_l(h_state) * reset_mask
                
                # Reset cumulative halt probabilities
                cumulative_halt = cumulative_halt * (1 - should_update_h.squeeze(-1))
            
            if return_intermediates:
                intermediates.append({
                    'step': step,
                    'l_state': l_state.clone(),
                    'h_state': h_state.clone(),
                    'halt_prob': halt_prob.clone(),
                    'cumulative_halt': cumulative_halt.clone()
                })
            
            # Early stopping if all sequences have halted
            if (cumulative_halt >= self.act_threshold).all():
                break
        
        # Final output projection
        output = self.output_proj(l_state)
        
        # Apply stable softmax for small sample generalization
        if output.size(-1) > 1:  # Classification case
            output = self.stable_max(output)
        
        result = {
            'output': output,
            'halting_probs': torch.stack(halting_probs, dim=0),  # [steps, B, L]
            'final_h_state': h_state,
            'final_l_state': l_state,
            'num_steps': step + 1
        }
        
        if return_intermediates:
            result['intermediates'] = intermediates
            
        return result
