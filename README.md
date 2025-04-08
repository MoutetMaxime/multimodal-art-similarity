# README for multimodal-art-similarity

# 🎨 multimodal-art-similarity

A research project aimed at identifying stolen artworks by automatically comparing their descriptions and images to those of artworks exhibited in French museums.

## 🔍 Objective

Develop a **multimodal similarity search system** (text + image) to:
- Match **stolen artwork descriptions** (usually in text and/or image form)
- With **museum collection data** (titles, metadata, images)
- And detect potential correspondences.

## 🧠 Models

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual sentence embeddings)
- `xlm-roberta-large` (optional, for stronger multilingual performance)
- `CLIP` (optional, for joint text-image embeddings)
- Additional models to explore: BLIP, ALIGN, Florence, etc.

## 📁 Project Structure

```bash
multimodal-art-similarity/
├── data/
│   ├── stolen_art_db.csv              # Database of stolen artworks
│   ├── museum_collections.csv         # Database of museum artworks
│   └── images/                        # Artwork images (optional)
├── models/
│   └── embedding_model.py             # Text/image embedding utilities
├── search/
│   └── similarity_search.py           # FAISS or cosine-based search
├── notebooks/
│   └── 01_data_exploration.ipynb      # Initial data analysis
├── app/
│   └── streamlit_app.py               # Optional search interface
├── README.md
└── requirements.txt
```

## 🚀 Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/your_username/multimodal-art-similarity.git
   cd multimodal-art-similarity
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run a basic similarity search:
   ```bash
   python search/similarity_search.py
   ```

## 📚 References

## ✍️ Author

**Maxime Moutet**  [X / ENSAE]
