# pip install pandas numpy scikit-learn joblib
from __future__ import annotations
import os, math, joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, alpha: float = 0.5):
        # alpha: weight for content-based (1-alpha for collaborative)
        self.alpha = alpha
        self.items = None
        self.ratings = None

        # content-based
        self.vectorizer = None
        self.item_tfidf = None

        # collaborative
        self.user_index = {}
        self.item_index = {}
        self.R = None
        self.S = None

    # FIT
    def fit(self, items: pd.DataFrame, ratings: pd.DataFrame):
        '''
        items: DataFrame[item_id, title, tags, description]
        ratings: DataFrame[user_id, item_id, rating
        '''
        self.items = items.copy()
        self.ratings = ratings.copy()

        # content-based
        corpus = (
            self.items["title"] + " " +
            self.items["tags"] + " " +
            self.items["description"]
        ).values
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.item_tfidf = self.vectorizer.fit_transform(corpus)

        # collaborative (item-based CF)
        unique_users = self.ratings["user_id"].unique().tolist()
        unique_items = self.items["item_id"].unique().tolist()
        self.user_index = {u:i for i,u in enumerate(unique_users)}
        self.item_index = {it:i for i,it in enumerate(unique_items)}

        self.R = np.zeros((len(unique_users), len(unique_items)))
        for _, row in self.ratings.iterrows():
            ui = self.user_index[row.user_id]
            ii = self.item_index[row.item_id]
            self.R[ui, ii] = row.rating

        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-9
        Rn = self.R / norms
        self.S = Rn.T @ Rn
        np.fill_diagonal(self.S, 0.0)
        return self

    # PREDICT
    def predict(self, user_id: int, k: int = 5) -> List[Tuple[int,float]]:
        if user_id not in self.user_index:
            return []

        # content-based scores
        cb_scores = self._cb_scores(user_id)

        # collaborative scores
        cf_scores = self._cf_scores(user_id)

        # hybrid blend
        return self._hybrid(cb_scores, cf_scores, k)

    def _cb_scores(self, user_id: int) -> List[Tuple[int,float]]:
        user_rates = self.ratings[(self.ratings.user_id==user_id) & (self.ratings.rating>=4)]
        if user_rates.empty:
            return []
        liked_ids = user_rates["item_id"].tolist()
        liked_idx = self.items.index[self.items["item_id"].isin(liked_ids)].tolist()
        user_vec = self.item_tfidf[liked_idx].mean(axis=0)
        sims = cosine_similarity(user_vec, self.item_tfidf)[0]

        viewed = set(self.ratings[self.ratings.user_id==user_id]["item_id"].tolist())
        return [(int(self.items.iloc[i]["item_id"]), float(sims[i]))
                for i in np.argsort(sims)[::-1] if int(self.items.iloc[i]["item_id"]) not in viewed]

    def _cf_scores(self, user_id: int) -> List[Tuple[int,float]]:
        ui = self.user_index[user_id]
        user_ratings = self.R[ui]
        scores = self.S @ user_ratings
        norm = np.abs(self.S).sum(axis=1) + 1e-9
        preds = scores / norm
        preds[np.where(user_ratings > 0)] = -np.inf
        inv_item_index = {v:k_ for k_,v in self.item_index.items()}
        return [(int(inv_item_index[i]), float(preds[i])) for i in np.argsort(preds)[::-1] if preds[i] != -np.inf]

    def _hybrid(self, cb, cf, k) -> List[Tuple[int,float]]:
        def normalize(sc):
            if not sc:
                return {}
            vals = np.array([s for _,s in sc])
            vmin, vmax = vals.min(), vals.max()
            if math.isclose(vmax, vmin):
                return {i: 1.0 for i,_ in sc}
            return {i: float((s - vmin)/(vmax - vmin)) for i,s in sc}

        cdict = normalize(cb)
        fdict = normalize(cf)
        keys = set(cdict.keys()) | set(fdict.keys())
        blended = []
        for i in keys:
            blended.append((i, self.alpha*cdict.get(i,0.0)+(1-self.alpha)*fdict.get(i,0.0)))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:k]

    # SAVE/LOAD 
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HybridRecommender":
        return joblib.load(path)


# Evaluation 
def precision_at_k(preds: List[int], ground_truth: List[int], k: int) -> float:
    if not preds: return 0.0
    preds_k = preds[:k]
    relevant = set(ground_truth)
    hits = sum([1 for p in preds_k if p in relevant])
    return hits / k

def average_precision(preds: List[int], ground_truth: List[int], k: int) -> float:
    relevant = set(ground_truth)
    score = 0.0
    hits = 0
    for i, p in enumerate(preds[:k], start=1):
        if p in relevant:
            hits += 1
            score += hits / i
    if not relevant: return 0.0
    return score / min(len(relevant), k)

def map_at_k(all_preds: Dict[int,List[int]], all_truth: Dict[int,List[int]], k: int) -> float:
    aps = []
    for uid in all_preds:
        aps.append(average_precision(all_preds[uid], all_truth.get(uid,[]), k))
    return np.mean(aps) if aps else 0.0
