import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class EmbeddingFromPretrained:
    def __init__(self, model_name: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.emb_size = self.model.config.hidden_size
    
    def get_cls_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :]
    
    def get_mean_pooling_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ImageEmbeddingFromPretrained:
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.emb_size = self.model.config.hidden_size

    def get_mean_pooling_embedding(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state  # (1, seq_len, dim)
        embedding = token_embeddings.mean(dim=1)  # (1, dim)

        # L2 normalization
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding
