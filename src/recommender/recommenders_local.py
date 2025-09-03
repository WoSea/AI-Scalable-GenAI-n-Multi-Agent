# pip install pandas numpy scikit-learn
# pip install langchain-community

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langchain_community.chat_models import ChatOllama
    HAVE_OLLAMA = True
except Exception:
    HAVE_OLLAMA = False


# Fake data
def build_demo_data():
    # Items: id, title, tags, description
    items = pd.DataFrame([
        (1, "Interstellar", "sci-fi space drama", "A space exploration epic about love, time, and survival."),
        (2, "Inception", "sci-fi heist mind-bending", "A thief steals secrets using dream-sharing technology."),
        (3, "The Dark Knight", "action crime hero", "A hero faces an anarchist who plunges the city into chaos."),
        (4, "Gravity", "sci-fi space survival", "Two astronauts struggle to survive after a space accident."),
        (5, "La La Land", "romance music drama", "A love story between a jazz musician and an actress."),
        (6, "Whiplash", "music drama intensity", "A driven jazz drummer and an abusive instructor clash."),
        (7, "The Martian", "sci-fi space survival", "An astronaut stranded on Mars fights to stay alive."),
        (8, "Arrival", "sci-fi linguistics first-contact", "A linguist communicates with extraterrestrials."),
        (9, "Blade Runner 2049", "sci-fi neo-noir ai", "A blade runner unearths a long-buried secret."),
        (10, "Her", "romance ai introspective", "A man develops a relationship with an operating system."),
    ], columns=["item_id","title","tags","description"])

    # User interactions: user_id, item_id, rating (1~5)
    ratings = pd.DataFrame([
        (101, 1, 5), (101, 2, 5), (101, 3, 4),
        (102, 5, 5), (102, 6, 4), (102, 2, 3),
        (103, 7, 5), (103, 4, 4), (103, 8, 5),
        (104, 9, 4), (104,10, 5), (104, 2, 4),
        (105, 3, 5), (105, 1, 4), (105, 9, 5),
    ], columns=["user_id","item_id","rating"])
    return items, ratings

# CONTENT-BASED FILTERING (TF-IDF + cosine)
@dataclass
class ContentBasedRecommender:
    items: pd.DataFrame
    vectorizer: TfidfVectorizer = None
    item_tfidf: np.ndarray = None
    '''
    def tf(self, term, doc):
        result = 0
        for word in doc:
            if word == term:
                result += 1
        return result / len(doc)
    def idf(self, term, docs):
        result = 0
        for doc in docs:
            for word in doc:
                if word == term:
                    result += 1
                    break
        return math.log(len(docs) / result, math.e) if result > 0 else 0.0
    def tf_idf(self, term, doc, docs):
        return self.tf(term, doc) * self.idf(term, docs)
    '''
    def fit(self):
        corpus = (self.items["title"] + " " +
                  self.items["tags"] + " " +
                  self.items["description"]).values
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.item_tfidf = self.vectorizer.fit_transform(corpus)  # (N_items x V)
        return self

    def similar_items(self, item_id: int, k: int = 5) -> List[Tuple[int, float]]:
        idx = self.items.index[self.items["item_id"] == item_id][0]
        sims = cosine_similarity(self.item_tfidf[idx], self.item_tfidf)[0]  # (N_items,)
        # self removed
        sims[idx] = -1.0
        top_idx = sims.argsort()[::-1][:k]
        return [(int(self.items.iloc[i]["item_id"]), float(sims[i])) for i in top_idx]

    def recommend_for_user(self, ratings: pd.DataFrame, user_id: int, k: int = 5) -> List[Tuple[int, float]]:
        # Calculate "user profile vector" from items liked by user (rating >= 4)
        user_rates = ratings[(ratings.user_id==user_id) & (ratings.rating>=4)]
        if user_rates.empty:
            return []
        liked_ids = user_rates["item_id"].tolist()
        liked_idx = self.items.index[self.items["item_id"].isin(liked_ids)].tolist()
        user_vec = self.item_tfidf[liked_idx].mean(axis=0)  # average
        sims = cosine_similarity(user_vec, self.item_tfidf)[0]

        # filter out already viewed items
        viewed = set(ratings[ratings.user_id==user_id]["item_id"].tolist())
        candidates = [(int(self.items.iloc[i]["item_id"]), float(sims[i]))
                      for i in np.argsort(sims)[::-1] if int(self.items.iloc[i]["item_id"]) not in viewed]
        return candidates[:k]


# COLLABORATIVE FILTERING (item-based, cosine on rows)
@dataclass
class ItemBasedCF:
    items: pd.DataFrame
    ratings: pd.DataFrame
    item_index: Dict[int,int] = None
    user_index: Dict[int,int] = None
    R: np.ndarray = None           # user-item matrix (num_users x num_items)
    S: np.ndarray = None           # item-item similarity

    def fit(self):
        unique_users = self.ratings["user_id"].unique().tolist()
        unique_items = self.items["item_id"].unique().tolist()
        self.user_index = {u:i for i,u in enumerate(unique_users)}
        self.item_index = {it:i for i,it in enumerate(unique_items)}

        self.R = np.zeros((len(unique_users), len(unique_items)), dtype=float)
        for _, row in self.ratings.iterrows():
            ui = self.user_index[row.user_id]
            ii = self.item_index[row.item_id]
            self.R[ui, ii] = row.rating

        # cosine sim between vectors ratings of item
        # 1e-9 to avoid division by zero
        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-9
        Rn = self.R / norms
        self.S = Rn.T @ Rn  # item-item similarity
        np.fill_diagonal(self.S, 0.0)
        return self

    def recommend_for_user(self, user_id: int, k: int = 5) -> List[Tuple[int, float]]:
        if user_id not in self.user_index:
            return []
        ui = self.user_index[user_id]
        user_ratings = self.R[ui]  # (num_items,)
        # Predict score for item j: sum(sim(j, i)*rating_i)/sum|sim|
        scores = self.S @ user_ratings
        norm = np.abs(self.S).sum(axis=1) + 1e-9
        preds = scores / norm

        # filter out already viewed items
        seen = np.where(user_ratings > 0)[0]
        preds[seen] = -np.inf

        top_idx = np.argsort(preds)[::-1][:k]
        inv_item_index = {v:k_ for k_,v in self.item_index.items()}
        return [(int(inv_item_index[i]), float(preds[i])) for i in top_idx if preds[i] != -np.inf]


# HYBRID = alpha * content + (1-alpha) * CF
def hybrid_recommend(
    content_scores: List[Tuple[int, float]],
    cf_scores: List[Tuple[int, float]],
    alpha: float = 0.5,
    k: int = 5
) -> List[Tuple[int, float]]:
    # Normalize scores for each branch (min-max) then blend
    def normalize(sc):
        if not sc:
            return {}
        vals = np.array([s for _,s in sc], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if math.isclose(vmax, vmin):
            return {i: 1.0 for i,_ in sc}
        return {i: float((s - vmin)/(vmax - vmin)) for i,s in sc}

    cdict = normalize(content_scores)
    fdict = normalize(cf_scores)
    keys = set(cdict.keys()) | set(fdict.keys())
    blended = []
    for i in keys:
        c = cdict.get(i, 0.0)
        f = fdict.get(i, 0.0)
        blended.append((i, alpha*c + (1.0-alpha)*f))
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended[:k]


# Explain recommendations by LLM (Ollama-mistral)
def explain_with_llm(items_df: pd.DataFrame, recs: List[Tuple[int,float]], user_profile_text: str = "") -> str:
    if not HAVE_OLLAMA:
        return "(LLM explainer skipped — install & run Ollama to enable.)"
    try:
        llm = ChatOllama(model="mistral", temperature=0.2)
        titles = [items_df.loc[items_df.item_id==i, "title"].values[0] for i,_ in recs]
        prompt = (
            "As a movie recommendation assistant, "
            "explain why the user might like the following movies, based on similar genre/theme descriptions:\n"
            f"User profile: {user_profile_text}\n"
            f"Recommended movies: {', '.join(titles)}\n"
            "Provide a brief explanation in bullet points, avoiding vague statements."
        )
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return f"(LLM explainer error: {e})"

if __name__ == "__main__":
    items, ratings = build_demo_data()

    # Content-based
    cb = ContentBasedRecommender(items).fit()
    user_id = 101
    cb_recs = cb.recommend_for_user(ratings, user_id=user_id, k=5)

    # Collaborative (item-based)
    cf = ItemBasedCF(items, ratings).fit()
    cf_recs = cf.recommend_for_user(user_id=user_id, k=5)

    # Hybrid
    hyb_recs = hybrid_recommend(cb_recs, cf_recs, alpha=0.6, k=5)

    # Print output
    def pretty(name, recs):
        df = pd.DataFrame(recs, columns=["item_id","score"])
        df = df.merge(items[["item_id","title","tags"]], on="item_id", how="left")
        print(f"\n {name} Recommendations for user {user_id}")
        print(df.to_string(index=False))

    pretty("Content-Based", cb_recs)
    pretty("Collaborative (Item-Based CF)", cf_recs)
    pretty("Hybrid (α=0.6*CB + 0.4*CF)", hyb_recs)

    # Explain by LLM (Ollama-mistral) why user 101 might like these movies
    # As a movie recommendation assistant, explain why the user might like the following movies, based on similar genre/theme descriptions
    # Create user profile summary from highly rated movies
    liked = ratings[(ratings.user_id==user_id) & (ratings.rating>=4)] \
                .merge(items, on="item_id") \
                [["title","tags"]]
    profile_text = "; ".join([f"{r.title} ({r.tags})" for r in liked.itertuples(index=False)])
    explanation = explain_with_llm(items, hyb_recs, user_profile_text=profile_text)
    print("\n LLM Explanation")
    print(explanation)


'''
 Content-Based Recommendations for user 101
 item_id     score              title                      tags
       4 0.52135248            Gravity     sci-fi space survival
       7 0.49105218        The Martian     sci-fi space survival
       8 0.45611367           Arrival sci-fi linguistics first-contact
       9 0.44392731 Blade Runner 2049          sci-fi neo-noir ai
      10 0.37288545               Her        romance ai introspective

 Collaborative (Item-Based CF) Recommendations for user 101
 item_id     score              title                      tags
       7 4.59892893        The Martian     sci-fi space survival
       4 4.40853285            Gravity     sci-fi space survival
       8 3.21787709           Arrival sci-fi linguistics first-contact
       9 2.98234578 Blade Runner 2049          sci-fi neo-noir ai
      10 2.11267549               Her        romance ai introspective

 Hybrid (α=0.6*CB + 0.4*CF) Recommendations for user 101
 item_id     score              title                      tags
       7 0.96231125        The Martian     sci-fi space survival
       4 0.95202837            Gravity     sci-fi space survival
       8 0.67944551           Arrival sci-fi linguistics first-contact
       9 0.65087673 Blade Runner 2049          sci-fi neo-noir ai
      10 0.48133838               Her        romance ai introspective

 LLM Explanation 
 **The Martian**: You enjoyed *Interstellar* both are science fiction films about space, survival, and humanity’s hope in extreme conditions.

 **Gravity**: Similar to *Interstellar* in its focus on space and survival tension. If you liked the stunning outer space visuals, *Gravity* is a perfect match.

 **Arrival**: Since you appreciated *Inception* for its intellectual depth and “mind-bending” narrative, *Arrival* offers a similarly thought-provoking experience, this time centered on language and the perception of time.

 **Blade Runner 2049**: Much like *The Dark Knight*, this film explores complex characters, a noir atmosphere, and ethical questions about humanity and artificial intelligence.

 **Her**: Given your interest in psychological and technology-driven themes (*Inception*, *Blade Runner*), *Her* provides a more emotional angle – the human-AI relationship, deeply reflective and humanistic.

 Overall, the system recommends movies that blend **space survival sci-fi** (*The Martian*, *Gravity*), **deep intellectual ideas** (*Arrival*, *Blade Runner 2049*), and **emotional, humanistic depth** (*Her*), aligning closely with your tastes shown in *Interstellar* and *Inception*.
'''