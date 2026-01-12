import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_patient_gene_embeddings(
    path,
    device="cpu"
):
    """
    Load patient and gene embeddings from a .pt file.

    Args:
        path (str): Path to the .pt file
        patient_key (str): Key for patient embeddings in each dict
        gene_key (str): Key for gene embeddings in each dict
        device (str or torch.device): Device to move tensors to

    Returns:
        tuple: (patient_embs, gene_embs) as torch.Tensors
    """
    data = torch.load(path, map_location="cpu")
    patient_key = "patient"
    gene_key = "causal_gene_enc"

    assert isinstance(data, list), "Expected a list of dictionaries"

    patient_embs = []
    gene_embs = []

    for item in data:
        patient_embs.append(item[patient_key])
        gene_embs.append(item[gene_key])

    patient_embs = torch.stack(patient_embs).to(device)
    gene_embs = torch.stack(gene_embs).to(device)

    print(f"Loaded embeddings from: {path}")
    print(f" - Patient Embs: {patient_embs.shape}")
    print(f" - Gene Embs: {gene_embs.shape}\n")

    return patient_embs, gene_embs



# DATA_PATH = "embs/patients_train.pt"

# patient_embs, gene_embs = load_patient_gene_embeddings(
#     path=DATA_PATH,
#     device=device
# )

# print(len(patient_embs), len(gene_embs))