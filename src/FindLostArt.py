import gc
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from Embedding import EmbeddingFromPretrained, ImageEmbeddingFromPretrained
from utils.download import download_image_in_memory, extract_slider_image_urls
from utils.processing_df import (
    add_column_with_concatenated_txt,
    find_lostart_csv,
    find_lostart_csvs,
    get_concatenated_txt,
    keep_necessary_columns_la,
    keep_necessary_columns_mnr,
)
from utils.timing import timing


class FindLostArt:
    def __init__(
            self,
            start: int=0,
            language_model: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cls_embedding: bool=True,
            vision_model: str="facebook/dinov2-base"
        ):

        self.lostart = pd.read_csv(f"data/lostart/lostart_start={start}.csv", sep=";")
        self.mnr = pd.read_excel("data/mnr_20250303_17h40m54s.ods")

        self.found = pd.read_csv("data/found_lostart.csv")
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
        # self.mnr = remove_leakage_mnr(self.mnr)

        # Create CONCATENATED column in MNR
        self.mnr = add_column_with_concatenated_txt(self.mnr)

        # Chose embedding
        embedder = EmbeddingFromPretrained(model_name=language_model)
        self.embedding = embedder.get_cls_embedding if cls_embedding else embedder.get_mean_pooling_embedding
        self.emb_size = embedder.emb_size

        self.vision_model = vision_model


    @timing
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
        if top_n is None:
            top_n = len(self.mnr)
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values

    @timing
    def get_most_similar_image(self, emb: torch.tensor, vision_embedder: Callable[[Image.Image], torch.tensor], top_n: Optional[int]=None):
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
        
        Returns
        -------
        pd.DataFrame
            The most similar images.
        np.ndarray
            The similarity scores.
        """
        results = []

        # We stream the images from their urls
        for mnr, mnr_url in self.mnr[["REF", "VIDEO"]].values:
            # Get the image urls
            slider_image_urls = extract_slider_image_urls(mnr_url)

            # We use only the first image
            if slider_image_urls:
                mnr_im = slider_image_urls[0]
                mnr_im = download_image_in_memory(mnr_im)

                # Compute embedding
                mnr_emb = vision_embedder(mnr_im)

                # We delete the image to free memory
                del mnr_im

                # Compute similarity
                similarities = cosine_similarity(emb, mnr_emb).item()

                results.append({"REF": mnr_url, "similarity_image": similarities})


        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="similarity_image", ascending=False)
        if top_n is None:
            top_n = len(df_results)
        return df_results.head(top_n), df_results["similarity_image"].head(top_n).values


    @timing
    def get_most_similar_title_author_desc(self, embs: List[torch.tensor], top_n: int=None, agg: str="mean"):
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
        agg : str
            The aggregation method to use. Can be "mean" or "max".

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """
        # Compute the similarity for each column
        # The cols should be ordered in the same way as the mnr dataframe
        sims = pd.DataFrame()
        for i, mnr_col in enumerate(self.mnr2la.keys()):
            if embs[i] is not None:
                sims[mnr_col] = self.mnr[mnr_col].astype(str).apply(lambda x: cosine_similarity(embs[i], self.embedding(x)).item())

        # Compute the agg of the similarities
        if agg == "mean":
            similarities = np.mean(sims, axis=1)
        elif agg == "max":
            similarities = np.max(sims, axis=1)
        else:
            raise ValueError(f"Aggregation method {agg} not supported. Use 'mean' or 'max'.")
        
        # Get the most similar texts
        if top_n is None:
            top_n = len(self.mnr)
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values


    def calculate_similarity(row: pd.Series, beta_default: int=0.5):
        if pd.isna(row["similarity_image"]):  # Si l'image est absente
            beta = 1.0  # On ne prend en compte que le texte
        else:
            beta = beta_default  # Valeur de beta par défaut, lorsque l'image est présente

        # Calcul final de la similarité
        return beta * row["similarity_text"] + (1 - beta) * row["similarity_image"]

    def search_lostart(self, id: int, top_n: int=5, cross_check: bool=False, use_vision: bool=False, beta: float=0.5):
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

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """
        if id not in self.lostart["Lost Art ID"].values:
            csv, df = find_lostart_csv(id)
            raise ValueError(f"Database not initialized properly. Lost Art ID {id} not in database but found in {csv} file.")


        df = self.lostart.loc[self.lostart["Lost Art ID"] == id]
        df = df.drop(columns=["Lost Art ID"]).squeeze()

        if not cross_check:
            # We compute the similarity of the Lost Art with a concatenated string of all columns from mnr
            # Concatenate in one string
            txt = get_concatenated_txt(df)

            # Get the embedding
            embedding = self.embedding(txt)

            # Search in MNR
            similar_text, similarities = self.get_most_similar_text(embedding)
            similar_text = similar_text.drop(columns=["CONCATENATED"])
        
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
            similar_text, similarities = self.get_most_similar_title_author_desc(embs, agg="mean")
            similar_text = similar_text.drop(columns=["CONCATENATED"])
            similar_text["similarity_text"] = similarities
        
        if use_vision:
            vision_embedder = ImageEmbeddingFromPretrained(model_name=self.vision_model).get_mean_pooling_embedding

            # Get the image embedding
            image_path = os.path.join("data/images/lostart", f"{id}.jpg")
            if os.path.exists(image_path):
                image_embedding = vision_embedder(image_path)
                similar_image, similarities_image = self.get_most_similar_image(image_embedding, vision_embedder)
        
            # Merge the two dataframes
            similar_art = pd.merge(similar_text, similar_image, on="REF", how="left")
            similar_art = similar_text.rename(columns={"similarity_x": "similarity_text", "similarity_y": "similarity_image"})

            # Compute the final similarity
            similar_art["similarity"] = similar_art.apply(self.calculate_similarity, axis=1)
            similar_art = similar_art.drop(columns=["similarity_text", "similarity_image"])
        else:
            # We only keep the similarity of the text
            similar_art = similar_text.rename(columns={"similarity": "similarity_text"})

        # Sort the dataframe by similarity
        similar_art = similar_art.sort_values(by="similarity", ascending=False)
        if top_n is not None:
            similar_art = similar_art.head(top_n)
        return similar_art.reset_index(drop=True), similarities[:top_n].tolist()
    

    def evaluate_on_found(self, top_n: int=10, cross_check: bool=False):
        """
        Evaluate the model on the found Lost Art IDs.
        """
        result = pd.DataFrame(columns=["Lost Art ID", "rank", "similarity"])

        for i, id in enumerate(self.found["Lost Art ID"]):
            print(f"Searching for Lost Art ID {id} in MNR...")
            if id not in self.lostart["Lost Art ID"].values:
                # Add the row to the lostart dataframe
                self.lostart = pd.concat([self.lostart, self.found.loc[[i]]], ignore_index=True)

            similar_text, similarities = self.search_lostart(id, top_n=top_n, cross_check=cross_check)

            # Check if we found the Lost Art ID in MNR
            if self.found2found[id] in similar_text["REF"].values:
                rank = similar_text.loc[similar_text["REF"] == self.found2found[id]].index[0]
                sim = similarities[rank]
            else:
                rank = -1
                sim = -1

            # Add the rank and similarity score to the dataframe
            result = pd.concat([result, pd.DataFrame({"Lost Art ID": [id], "rank": [rank], "similarity": [sim]})], ignore_index=True)
            print(f"Lost Art ID {id} found in MNR with rank {rank} and similarity {sim}")

        return result


if __name__ == "__main__":
    timing.ENABLE_TIMING = True
    # find_lostart = FindLostArt()
    
    # Search for a Lost Art ID
    # res = find_lostart.search_lostart(589707, top_n=5, cross_check=False)
    # print(res)

    print(find_lostart_csvs([310418]))
