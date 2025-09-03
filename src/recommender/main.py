import pandas as pd
from recommender_module import HybridRecommender, precision_at_k, map_at_k

# Fake data
items = pd.DataFrame([
    (1,"Interstellar","sci-fi","Space epic"),
    (2,"La La Land","romance","Musical love story"),
    (3,"The Dark Knight","action","Hero vs villain"),
    (4,"The Martian","sci-fi","Astronaut survival"),
], columns=["item_id","title","tags","description"])

ratings = pd.DataFrame([
    (101,1,5),(101,3,4),
    (102,2,5),(102,4,4),
], columns=["user_id","item_id","rating"])

# Fit
rec = HybridRecommender(alpha=0.6).fit(items, ratings)

# Predict for user 101
recs = rec.predict(101, k=3)
print("Recommendations:", recs)

# Evaluate (Precision@K, MAP@K)
all_preds = {101: [i for i,_ in rec.predict(101, k=3)]}
all_truth = {101: [4]}  # suppose item 4 is relevant
print("P@3:", precision_at_k(all_preds[101], all_truth[101], 3))
print("MAP@3:", map_at_k(all_preds, all_truth, 3))

# Save / Load
rec.save("recommender.pkl")
rec2 = HybridRecommender.load("recommender.pkl")
print("Reloaded rec:", rec2.predict(102, k=3))
