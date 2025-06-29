import torch
from simcse_model.SimCSE import SimCSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_simcse(model_path="simcse_minilm.pt", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("Loaded model ✅")
    model = SimCSE(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    def embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            embeddings = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]) #will indirectly call forward func
            return embeddings.cpu().numpy()
    
    return embed
