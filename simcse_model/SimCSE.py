from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Example: Using SNLI for open-domain sentence pairs
dataset = load_dataset("snli", split="train")
sentences = list(set([x["premise"] for x in dataset if x["premise"] is not None]))
sentences = sentences[:10000]  # Optional: truncate for fast testing

class SimCSE(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, **kwargs):
        output = self.encoder(**kwargs)
        return output.last_hidden_state[:, 0]  # CLS token
                                                #output.last_hidden_state has shape (batch_size, sequence_length, x).
                                            #[:, 0] slices out the CLS token representation from each sentence — 
                                            # this becomes the sentence embedding.
    
def get_positive_pairs(batch, tokenizer, device):
    augmented = [s + "." for s in batch]  # simple dropout trigger
    encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    encoded2 = tokenizer(augmented, padding=True, truncation=True, return_tensors="pt").to(device)

    emb1 = model(**encoded)     # Keep gradient tracking ON
    emb2 = model(**encoded2)

    return emb1, emb2

    
def simcse_loss(emb1, emb2, temperature=0.05):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    similarities = emb1 @ emb2.T  # Cosine similarities
    labels = torch.arange(emb1.size(0)).to(emb1.device)
    loss = F.cross_entropy(similarities / temperature, labels)
    return loss

if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimCSE("sentence-transformers/all-MiniLM-L6-v2").to(device)  # or whatever base model you're using

    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    BATCH_SIZE = 64
    EPOCHS = 3

    for epoch in range(EPOCHS):
        random.shuffle(sentences)
        dataloader = DataLoader(sentences, batch_size=BATCH_SIZE)

        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            emb1, emb2 = get_positive_pairs(batch, tokenizer, device)
            loss = simcse_loss(emb1, emb2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        
    # Save model weights
    torch.save(model.state_dict(), "simcse_minilm.pt")
    print("✅ Model saved to simcse_minilm.pt")
