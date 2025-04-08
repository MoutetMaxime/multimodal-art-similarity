import torch
from transformers import AutoModel, AutoTokenizer


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
