import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import utils.timing as timing
from Embedding import ImageEmbeddingFromPretrained, TextEmbeddingFromPretrained
from utils.image_tools import download_image_in_memory, extract_slider_image_urls
from utils.text_tools import (
    add_column_with_concatenated_txt,
    find_lostart_csv,
    find_lostart_csvs,
    get_concatenated_txt,
    keep_necessary_columns_la,
    keep_necessary_columns_mnr,
)


class FindLostArt:
    def __init__(
            self,
            start: int=0,
            language_model: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cls_embedding: bool=True,
            vision_model: str="facebook/dinov2-base",
            device: Optional[str] = None
        ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Load the data
        self.lostart = pd.read_csv(os.path.join(self.base_dir, "data", "lostart", f"lostart_start={start}.csv"), sep=";")
        self.mnr = pd.read_excel(os.path.join(self.base_dir, "data", "mnr_20250303.ods"))
        self.found = pd.read_csv(os.path.join(self.base_dir, "data", "found.csv"))

        self.found2found = {
            589707: "MNR00246",
            589708: "MNR00253",
            614072: "MNR00181",
            526702: "MNR00387",
            567247: "MNR00386",
            429210: "MNR00707",
            310418: "OAR00093",
            600027: "OAR00540",
            323038: "RFR00041"
        }

        self.mnr2la = {
            "AUTR": "Hersteller/Künstler/Autor:in",
            "TITR": "Titel",
            "REPR": "Beschreibung",
        }


        # Process the data
        self.lostart = keep_necessary_columns_la(self.lostart)
        self.found = keep_necessary_columns_la(self.found)
        self.mnr = keep_necessary_columns_mnr(self.mnr)

        # Create CONCATENATED column in MNR
        self.mnr = add_column_with_concatenated_txt(self.mnr)

        # Chose embedding
        embedder = TextEmbeddingFromPretrained(model_name=language_model, device=self.device)
        self.embedding = embedder.get_cls_embedding if cls_embedding else embedder.get_mean_pooling_embedding
        self.cls_embedding = cls_embedding
        self.emb_size = embedder.emb_size

        self.vision_model = vision_model


    @timing.timing
    def get_most_similar_text(self, emb: torch.tensor, top_n: int=None):
        """
        Get the most similar text to the given embedding in the given dataframe.
        Return the top_n most similar texts and their similarity scores.

        Parameters
        ----------
        emb : torch.tensor
            The embedding to compare with.
        top_n : int
            The number of most similar texts to return.
            If None, return all the texts.

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """
        similarities = self.mnr["CONCATENATED"].apply(lambda x: cosine_similarity(emb, self.embedding(x)).item())

        # Sort the dataframe by similarity
        similarities = similarities.sort_values(ascending=False)
        if top_n is None:
            top_n = len(self.mnr)
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values

    @timing.timing
    def get_most_similar_image(self, emb: torch.tensor, vision_embedder: Callable[[Image.Image], torch.tensor], top_n: Optional[int]=None, embeddings_file: Optional[str]=None):    
        """
        Get the most similar image to the given embedding in the given dataframe.
        Return the top_n most similar images and their similarity scores.

        Parameters
        ----------
        emb : torch.tensor
            The embedding to compare with.
        vision_embedder : Callable
            The function to use to compute the image embedding.
        top_n : int
            The number of most similar images to return.
            If None, return all the images.
        embeddings_file : str
            A precomputed embeddings file to use.
            If None, compute the embeddings from the dataframe.
        
        Returns
        -------
        pd.DataFrame
            The most similar images.
        np.ndarray
            The similarity scores.
        """
        results = []

        # Check if we already computed all embeddings (search for a .pt file in the data/embeddings folder)
        if embeddings_file is not None and os.path.exists(os.path.join(self.base_dir, "data", "embeddings", embeddings_file)):
            # We load the embeddings from the file
            mnr_emb = torch.load(os.path.join(self.base_dir, "data", "embeddings", embeddings_file))

            refs, embedding = mnr_emb["refs"], mnr_emb["embeddings"]

            similarities = cosine_similarity(emb, embedding.squeeze(1)).flatten()
            results = [{"REF": ref, "similarity_image": sim} for ref, sim in zip(refs, similarities)]

        else:
            # We stream the images from their urls
            for mnr in self.mnr["REF"].values:
                link = f"https://pop.culture.gouv.fr/notice/mnr/{mnr}"

                # Get the image urls
                # Each link has a slider with multiple images
                slider_image_urls = extract_slider_image_urls(link)

                # We use only the first image
                if slider_image_urls:
                    mnr_im = slider_image_urls[0]
                    mnr_im = download_image_in_memory(mnr_im)

                    # Compute embedding
                    mnr_emb = vision_embedder(mnr_im)

                    # We delete the image to free memory
                    del mnr_im

                    # Compute similarity
                    similaritiy = cosine_similarity(emb, mnr_emb).item()

                    results.append({"REF": mnr, "similarity_image": similaritiy})


        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="similarity_image", ascending=False)
        if top_n is None:
            top_n = len(df_results)
        return df_results.head(top_n), df_results["similarity_image"].head(top_n).values


    @timing.timing
    def get_most_similar_title_author_desc(self, embs: List[torch.tensor], top_n: int=None, embeddings_file: str=None):
        """
        Get the most similar text to the given embedding in the given dataframe.
        Return the top_n most similar texts and their similarity scores.

        Parameters
        ----------
        embs : torch.tensor
            The embeddings to compare with. Each row correspond to a column in the dataframe (ex: title, author, description).
        top_n : int
            The number of most similar texts to return.
            If None, return all the texts.
        embeddings_file : str
            A precomputed embeddings file to use.
            If None, compute the embeddings from the dataframe.

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """
        sims = pd.DataFrame()
        if embeddings_file is not None and os.path.exists(os.path.join(self.base_dir, "data", "embeddings", embeddings_file)):
            assert ("cls" in embeddings_file and self.cls_embedding) or ("mean_pooling" in embeddings_file and not self.cls_embedding), "The embeddings file does not match the embedding type."
            
            # We load the embeddings from the file
            mnr_emb = torch.load(os.path.join(self.base_dir, "data", "embeddings", embeddings_file))

            refs, embedding = mnr_emb["refs"], mnr_emb["embeddings"]
            for i, mnr_col in enumerate(self.mnr2la.keys()):
                if embs[i] is not None:
                    # Compute the similarity
                    sims[mnr_col] = cosine_similarity(embs[i], embedding[:, i, :]).flatten()

        else:
            # Compute the similarity for each column
            # The cols should be ordered in the same way as the mnr dataframe
            for i, mnr_col in enumerate(self.mnr2la.keys()):
                if embs[i] is not None:
                    sims[mnr_col] = self.mnr[mnr_col].astype(str).apply(lambda x: cosine_similarity(embs[i], self.embedding(x)).item())

        # Compute the mean of the similarities
        similarities = np.mean(sims, axis=1)
        
        # Get the most similar texts
        if top_n is None:
            top_n = len(self.mnr)
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values


    @staticmethod
    def calculate_similarity(row: pd.Series, beta_default: int=0.5):
        if pd.isna(row["similarity_image"]):
            return row["similarity_text"]
        else:
            return beta_default * row["similarity_text"] + (1 - beta_default) * row["similarity_image"]
    

    def rank_with_text(self, df: pd.Series, cross_comparison: bool=False, embeddings_file: str=None):
        """
        Compute the similarity of the Lost Art with the MNR dataframe and rank the results.

        Parameters
        ----------
        df : pd.Series
            The Lost Art to search for.
            Columns must include "CONCATENATED", "AUTR", "TITR", "REPR" and "Lost Art ID".
        cross_comparison : bool
            If True, compute the similarity of each column separately and return the average.
            If False, compute the similarity of the concatenated string of all columns.
            Default is False.
        Returns
        -------
        pd.DataFrame
            The most similar texts.
        """
        if not cross_comparison:
            # We compute the similarity of the Lost Art with a concatenated string of all columns from mnr
            # Concatenate in one string
            txt = get_concatenated_txt(df)

            # Get the embedding
            embedding = self.embedding(txt)

            # Search in MNR
            similar_text, similarities = self.get_most_similar_text(embedding)
            similar_text = similar_text.drop(columns=["CONCATENATED"])
            similar_text["similarity_text"] = similarities
        
        else:
            # We compute the similarity of the Lost Art title with the MNR title, the Lost Art author with the MNR author, etc.
            # We then return an average of the similarities

            # Compute the embedding for each column
            embs = []
            for col in self.mnr2la.values():
                if not pd.isna(df[col]):
                    embs.append(self.embedding(df[col]))
                else:
                    embs.append(None)

            # Compute the similarity for each column
            similar_text, similarities = self.get_most_similar_title_author_desc(embs, embeddings_file=embeddings_file)
            similar_text = similar_text.drop(columns=["CONCATENATED"])
            similar_text["similarity_text"] = similarities

        return similar_text


    def rank_with_image(self, id: int, embeddings_file: str=None):
        """
        Compute the similarity of the Lost Art with the MNR dataframe and rank the results.

        Parameters
        ----------
        id : int
            The Lost Art ID to search for.
        embeddings_file : str
            A precomputed embeddings file to use.
            If None, compute the embeddings from the dataframe.
        
        Returns
        -------
        pd.DataFrame
            The most similar images, in MNR.
        """
        vision_embedder = ImageEmbeddingFromPretrained(model_name=self.vision_model).get_cls_embedding

        # Get the image embedding
        image_path = os.path.join("data/images/lostart", f"{id}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_embedding = vision_embedder(image)
            similar_image, similarities_image = self.get_most_similar_image(image_embedding, vision_embedder, embeddings_file=embeddings_file)

        return similar_image



    def search_lostart(self, id: int, top_n: int=5, use_text:bool=True, use_vision: bool=False, cross_comparison: bool=False, beta: float=0.5, text_embeddings_file: str=None, image_embeddings_file: str=None):
        """
        Search the Lost Art ID in MNR with similarity search.

        Parameters
        ----------
        id : int
            The Lost Art ID to search for.
        top_n : int
            The number of most similar texts to return.
            If None, return all the texts.
        cross_check : bool
            If True, compute the similarity of each column separately and return the average.
            If False, compute the similarity of the concatenated string of all columns.
            Default is False.
        use_vision : bool
            If True, use the vision model to compute the similarity of the images.
            Default is False.
        beta : float
            The weight of the text similarity in the final similarity score.
            Default is 0.5.
        text_embeddings_file : str
            A precomputed embeddings file to use for the text similarity.
            If None, compute the embeddings from the dataframe.
        image_embeddings_file : str
            A precomputed embeddings file to use for the image similarity.
            If None, compute the embeddings from the dataframe.

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """
        assert use_text or use_vision, "At least one of use_text or use_vision must be True."

        if id not in self.lostart["Lost Art ID"].values:
            df, csv = find_lostart_csv(id)
            raise ValueError(f"Database not initialized properly. Lost Art ID {id} not in database but found in {csv[0]} file.")


        df = self.lostart.loc[self.lostart["Lost Art ID"] == id]
        df = df.drop(columns=["Lost Art ID"]).squeeze()

        # Compute similarity with text only
        if use_text:
            similar_text = self.rank_with_text(df, cross_comparison=cross_comparison, embeddings_file=text_embeddings_file)

        # Compute similarity with image only
        similar_image = None
        use_vision = use_vision and os.path.exists(os.path.join("data/images/lostart", f"{id}.jpg"))
        if use_vision:
            similar_image = self.rank_with_image(id, embeddings_file=image_embeddings_file)

        # If both are used, we merge the two dataframes
        if use_text and use_vision:
            # Merge the two dataframes
            similar_art = pd.merge(similar_text, similar_image, on="REF", how="left")
            similar_art = similar_art.rename(columns={"similarity_x": "similarity_text", "similarity_y": "similarity_image"})

            # Compute the final similarity
            similar_art["similarity"] = similar_art.apply(lambda row: self.calculate_similarity(row, beta_default=beta), axis=1)
            similar_art = similar_art.drop(columns=["similarity_text", "similarity_image"])
        else:
            if use_text:
                similar_art = similar_text.rename(columns={"similarity_text": "similarity"})
            else:
                if similar_image is not None:
                    similar_art = similar_image.rename(columns={"similarity_image": "similarity"})
                else:
                    similar_art = pd.DataFrame(columns=["REF", "similarity"])

        # Sort the dataframe by similarity
        similar_art = similar_art.sort_values(by="similarity", ascending=False)

        if top_n is not None:
            similar_art = similar_art.head(top_n)
        return similar_art.reset_index(drop=True)
    

    def evaluate_on_found(self, top_n: int=None, use_text: bool=True, use_vision: bool=True, cross_comparison: bool=False, beta: float=0.5, text_embeddings_file: str=None, image_embeddings_file: str=None):
        """
        Evaluate the model on the found Lost Art IDs.
        """
        result = {
            "Lost Art ID": [],
            "rank": [],
            "similarity": []
        }

        found, csvs = find_lostart_csvs(list(self.found2found.keys()))

        for i, id in enumerate(found["Lost Art ID"]):
            print(f"Searching for Lost Art ID {id} in MNR...")
            if id not in self.lostart["Lost Art ID"].values:
                # Add the row to the lostart dataframe
                self.lostart = pd.concat([self.lostart, found.iloc[[i]]])
            
            similar = self.search_lostart(id, top_n=top_n, use_text=use_text, cross_comparison=cross_comparison, use_vision=use_vision, beta=beta, text_embeddings_file=text_embeddings_file, image_embeddings_file=image_embeddings_file)

            # Check if we found the Lost Art ID in MNR
            if self.found2found[id] in similar["REF"].values:
                rank = similar.loc[similar["REF"] == self.found2found[id]].index[0]
                sim = similar.loc[similar["REF"] == self.found2found[id], "similarity"].values[0]
            else:
                rank = len(self.mnr)
                sim = len(self.mnr)

            # Add the rank and similarity score to the dataframe
            result["Lost Art ID"].append(id)
            result["rank"].append(rank)
            result["similarity"].append(sim)
            print(f"Lost Art ID {id} found in MNR with rank {rank} and similarity {sim}")

        return pd.DataFrame(result)


if __name__ == "__main__":
    timing.TimingConfig.ENABLE = False
    find_lostart = FindLostArt(start=8500, cls_embedding=False)
    text_embeddings_file = "mnr_text_minilm_mean_pooling_embeddings.pt"
    image_embeddings_file = "mnr_image_dino_cls_embeddings.pt"
    res = find_lostart.evaluate_on_found(cross_comparison=True, use_text=True, use_vision=True, beta=0.1, text_embeddings_file=text_embeddings_file, image_embeddings_file=image_embeddings_file)
    print(res)
