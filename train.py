import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from diffusers import DDPMScheduler

from models import FlatLatentDiffuser, sample_latent, train_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime
import torch
import matplotlib.pyplot as plt

def save_model(model, optimizer, losses, prefix="diffusion_model"):
    """
    Save the PyTorch model, optimizer, and losses with a timestamp.
    
    Args:
        model (torch.nn.Module): Trained model
        optimizer (torch.optim.Optimizer): Optimizer used
        losses (list[float]): Training loss history
        prefix (str): Prefix for the saved filename
    Returns:
        str: Saved filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, filename)
    
    print(f"\nModel saved to '{filename}'")
    return filename
def save_loss_plot(losses, prefix="training_loss"):
    """
    Save a training loss plot with a timestamp.
    
    Args:
        losses (list[float]): Training loss history
        prefix (str): Prefix for the saved filename
    Returns:
        str: Saved plot filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
    print(f"Training loss plot saved to '{filename}'")
    return filename
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",  # cosine schedule
    prediction_type="epsilon"
)
def train(model, patient_embs, gene_embs, epochs=100, batch_size=32):
    dataset = TensorDataset(patient_embs, gene_embs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    all_losses = []
    
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for z0, cond in loader:
            z0 = z0.to(device)
            cond = cond.to(device)
            loss = train_step(model, optimizer, scheduler, z0, cond, device)
            epoch_loss += loss * z0.size(0)
        
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")
        all_losses.append(epoch_loss)
    
    return all_losses




@torch.no_grad()
def generate_patients(model, scheduler, gene_embs, steps=50, z_dim=128):
    gene_embs = gene_embs.to(device)
    # Pass all required objects and the specific dimension
    z_gen = sample_latent(model, scheduler, gene_embs, steps=steps, z_dim=z_dim)
    return z_gen



def generate_mock_data(num_samples=100, patient_dim=128, gene_dim=256):
    """
    Generates random synthetic embeddings for testing purposes.
    
    Args:
        num_samples (int): Number of patient/gene pairs to create.
        patient_dim (int): Dimension of the patient latent space (z_dim).
        gene_dim (int): Dimension of the conditioning gene embeddings.
        
    Returns:
        tuple: (patient_embs, gene_embs) as torch.Tensors
    """
    # Random normal distribution is standard for latent embeddings
    patient_embs = torch.randn(num_samples, patient_dim)
    gene_embs = torch.randn(num_samples, gene_dim)
    
    print(f"Generated {num_samples} mock samples:")
    print(f" - Patient Embs: {patient_embs.shape}")
    print(f" - Gene Embs: {gene_embs.shape}\n")
    
    return patient_embs, gene_embs

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 500
    LEARNING_RATE = 1e-4
    
    patient_embs = None  # Load or generate patient embeddings here
    gene_embs = None     # Load or generate gene embeddings here

    patient_embs, gene_embs = generate_mock_data(
        num_samples=100, 
        patient_dim=256, 
        gene_dim=256
    )

    patient_dim = patient_embs.shape[1]
    gene_dim = gene_embs.shape[1]
    

    print("\nInitializing model...")
    model = FlatLatentDiffuser(
        z_dim=patient_dim,
        cond_dim=gene_dim,
        hidden_dim=512,
        time_dim=512
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    

    print("\nStarting training...")
    losses = train(model, patient_embs, gene_embs, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Save the model
    model_file = save_model(model, optimizer, losses)

    # Save the loss plot
    loss_plot_file = save_loss_plot(losses)


    
    # Generate samples for testing
    test_gene_embs = gene_embs[:10]
    generated_patients = generate_patients(
        model, 
        scheduler, 
        test_gene_embs, 
        steps=50, 
        z_dim=patient_dim  
    )

    print(f"\nGenerated patient embeddings shape: {generated_patients.shape}")
    print(f"Expected shape: torch.Size([{len(test_gene_embs)}, {patient_dim}])")

    # Compare with ground truth
    ground_truth = patient_embs[:10].to(device)
    mse = F.mse_loss(generated_patients, ground_truth)
    cosine_sim = F.cosine_similarity(generated_patients, ground_truth, dim=-1).mean()

    print(f"MSE with ground truth: {mse.item():.6f}")
    print(f"Average cosine similarity: {cosine_sim.item():.6f}")
    print("\nTraining and generation complete!")