from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from Embedding import EmbeddingFromPretrained
from utils.processing_df import (
    add_column_with_concatenated_txt,
    find_lostart_csv,
    get_concatenated_txt,
    keep_necessary_columns_la,
    keep_necessary_columns_mnr,
)
from utils.timing import timing


class FindLostArt:
    def __init__(
            self,
            start: int=0,
            model_name: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cls_embedding: bool=True
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
            # 310418: "OAR00093",
            # 600027: "OAR00540",
            # 323038: "RFR00041"
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
        embedder = EmbeddingFromPretrained(model_name=model_name)
        self.embedding = embedder.get_cls_embedding if cls_embedding else embedder.get_mean_pooling_embedding
        self.emb_size = embedder.emb_size
    
    @timing
    def get_most_similar_text(self, emb: torch.tensor, top_n: int=5):
        """
        Get the most similar text to the given embedding in the given dataframe.
        Return the top_n most similar texts and their similarity scores.

        Parameters
        ----------
        emb : torch.tensor
            The embedding to compare with.
        top_n : int
            The number of most similar texts to return.

        Returns
        -------
        pd.DataFrame
            The most similar texts.
        np.ndarray
            The similarity scores.
        """

        similarities = self.mnr["CONCATENATED"].apply(lambda x: cosine_similarity(emb, self.embedding(x)).item())
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values
    

    def get_most_similar_title_author_desc(self, embs: List[torch.tensor], top_n: int=5, agg: str="mean"):
        """
        Get the most similar text to the given embedding in the given dataframe.
        Return the top_n most similar texts and their similarity scores.

        Parameters
        ----------
        embs : torch.tensor
            The embeddings to compare with. Each row correspond to a column in the dataframe (ex: title, author, description).
        top_n : int
            The number of most similar texts to return.
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
        return self.mnr.loc[similarities.nlargest(top_n).index].reset_index(), similarities.nlargest(top_n).values


    def search_lostart(self, id: int, top_n: int=5, cross_check: bool=False):
        """
        Search the Lost Art ID in MNR with similarity search.
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
            similar_text, similarities = self.get_most_similar_text(embedding, top_n=top_n)
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
            similar_text, similarities = self.get_most_similar_title_author_desc(embs, top_n=top_n, agg="mean")
            similar_text = similar_text.drop(columns=["CONCATENATED"])

        return similar_text, similarities
    
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
    find_lostart = FindLostArt()
    
    # Search for a Lost Art ID
    res = find_lostart.search_lostart(589707, top_n=5, cross_check=False)
    print(res)