import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import DDPMScheduler

device = "cuda"

patient_dim = 128
gene_dim = 64
LATENT_SCALE = 0.18215  # from LDM / Stable Diffusion

# Positional Encoding for time steps
# Referance:
#            Attention is all you need: https://arxiv.org/abs/1706.03762
#            Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# A FiLM layer modulates a neural network’s features by applying a learned, feature-wise scale and shift 
# generated from a conditioning input, allowing the network to adapt its computation based on context.
# Referance:
#           FiLM: Visual Reasoning with a General Conditioning Layer: https://arxiv.org/abs/1709.07871
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, hidden_dim * 2)
    
    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class FlatLatentDiffuser(nn.Module):
    def __init__(self, z_dim, cond_dim, hidden_dim=512, time_dim=512):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Separate projections
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                'film': FiLMLayer(hidden_dim, hidden_dim) # cond_dim -> hidden_dim by cond_proj(cond)
            })
            for _ in range(4)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, z_dim)
    
    def forward(self, z_noisy, t, cond):
        t_emb = self.time_mlp(self.time_embed(t))
        
        # Combine z and time, condition via FiLM
        x = self.z_proj(z_noisy) + t_emb
        cond_emb = self.cond_proj(cond)
        
        for block in self.blocks:
            h = block['norm'](x)
            h = block['mlp'](h)
            h = block['film'](h, cond_emb)
            x = x + h
        
        return self.output_proj(x)


def train_step(model, optimizer, scheduler, z0, cond, device):
    model.train()
    optimizer.zero_grad()
    
    # 1. Sample random timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (z0.shape[0],), 
        device=device
    ).long()
    
    # 2. Add noise to the clean latents
    noise = torch.randn_like(z0)
    noisy_z = scheduler.add_noise(z0, noise, timesteps)
    
    # 3. Predict the noise and calculate loss
    noise_pred = model(noisy_z, timesteps, cond)
    loss = F.mse_loss(noise_pred, noise)
    
    # 4. Backprop
    loss.backward()
    optimizer.step()
    
    return loss.item()



@torch.no_grad()
def sample_latent(model, scheduler, cond, steps=50, z_dim=128):
    """
    Generates new patient embeddings.
    
    Args:
        model: The FlatLatentDiffuser instance
        scheduler: The DDPMScheduler instance
        cond: Gene embeddings [batch_size, gene_dim]
        steps: Number of denoising steps
        z_dim: The dimension of the patient latent space
    """
    model.eval()
    batch_size = cond.shape[0]
    device = cond.device
    
    # 1. Start from pure Gaussian noise with the correct shape
    # Shape: [batch_size, patient_dim]
    z = torch.randn((batch_size, z_dim), device=device)
    
    scheduler.set_timesteps(steps)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        # Predict noise residual
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model(z, timesteps, cond)
        
        # Compute the previous noisy sample (z_t -> z_t-1)
        z = scheduler.step(noise_pred, t, z).prev_sample
        
    return z
