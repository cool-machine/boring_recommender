"""
Item-to-Item Collaborative Filtering Implementation using Pandas
---------------------------------------------------------------

This script implements item-to-item collaborative filtering without requiring Spark.
It's designed to work with the same data format as the notebook but uses pure Pandas.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration
TRAIN_FP = "datasets/train_clicks.parquet"
VAL_FP = "datasets/valid_clicks.parquet"
NEIGHBORS_FP = "datasets/item_neighbors_pandas.parquet"
TOP_K_NEIGH = 50  # number of neighbors to keep per item

print("1. Loading data...")
# Load training data
if os.path.exists(TRAIN_FP):
    interactions = pd.read_parquet(TRAIN_FP)
    interactions = interactions.rename(columns={"click_article_id": "item_id"})
    interactions = interactions[["user_id", "item_id"]].drop_duplicates()
    
    # Print dataset statistics
    num_users = interactions['user_id'].nunique()
    num_items = interactions['item_id'].nunique()
    num_interactions = len(interactions)
    
    print(f"Dataset statistics:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Interactions: {num_interactions}")
    print(f"  Density: {num_interactions / (num_users * num_items):.6f}")
    
    # Load validation data
    if os.path.exists(VAL_FP):
        val = pd.read_parquet(VAL_FP)
        val = val.rename(columns={"click_article_id": "true_item"})
        val = val[["user_id", "true_item"]].drop_duplicates()
        
        val_users = val['user_id'].nunique()
        print(f"Validation users: {val_users}")
    else:
        print(f"Warning: Validation file {VAL_FP} not found.")
        val = pd.DataFrame(columns=["user_id", "true_item"])
else:
    print(f"Error: Training file {TRAIN_FP} not found.")
    exit(1)

print("\n2. Computing item popularity...")
# Compute item counts (popularity)
item_counts = interactions.groupby('item_id').size().reset_index(name='n_i')
print(f"Item counts statistics:")
print(item_counts['n_i'].describe())

print("\n3. Creating user-item matrix...")
# Create a sparse user-item matrix
user_ids = interactions['user_id'].unique()
item_ids = interactions['item_id'].unique()

# Create mappings for user and item indices
user_to_idx = {user: i for i, user in enumerate(user_ids)}
item_to_idx = {item: i for i, item in enumerate(item_ids)}
idx_to_item = {i: item for item, i in item_to_idx.items()}

# Create a sparse matrix
rows = [user_to_idx[user] for user in interactions['user_id']]
cols = [item_to_idx[item] for item in interactions['item_id']]
data = np.ones(len(interactions))

user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

print("\n4. Computing item similarity matrix...")
# Compute cosine similarity between items
# Note: This can be memory-intensive for large datasets
print("  Computing cosine similarity (this may take a while)...")
item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)

print("\n5. Extracting top neighbors for each item...")
# Extract top-K neighbors for each item
neighbors_list = []

for i in range(len(item_ids)):
    item = idx_to_item[i]
    # Get similarity scores for this item
    sim_scores = item_similarity[i].toarray().flatten()
    
    # Get indices of top neighbors (excluding self)
    sim_scores[i] = 0  # Exclude self-similarity
    top_indices = np.argsort(sim_scores)[-TOP_K_NEIGH:][::-1]
    top_scores = sim_scores[top_indices]
    
    # Filter out zero similarities
    nonzero_mask = top_scores > 0
    top_indices = top_indices[nonzero_mask]
    top_scores = top_scores[nonzero_mask]
    
    # Add to neighbors list
    for idx, score in zip(top_indices, top_scores):
        neighbors_list.append({
            'item_id': item,
            'neighbor_id': idx_to_item[idx],
            'sim': score
        })

# Convert to DataFrame
item_neighbors = pd.DataFrame(neighbors_list)

print(f"\nSample of item neighbors:")
print(item_neighbors.head(10))

# Count how many items have neighbors
items_with_neighbors = item_neighbors['item_id'].nunique()
print(f"Items with at least one neighbor: {items_with_neighbors} out of {num_items}")

print(f"\n6. Saving item neighbors to {NEIGHBORS_FP}...")
item_neighbors.to_parquet(NEIGHBORS_FP, index=False)

print("\n7. Generating recommendations...")
# Generate recommendations for users in validation set
if len(val) > 0:
    # Get user history from training data
    user_history = interactions.copy()
    
    # Generate recommendations
    print("  Computing recommendations...")
    
    # This is a simplified approach - for each user:
    # 1. Get their history
    # 2. Find similar items to those in their history
    # 3. Aggregate similarity scores
    # 4. Rank and recommend
    
    all_recs = []
    
    for user_id in val['user_id'].unique():
        # Get user's history
        user_items = interactions[interactions['user_id'] == user_id]['item_id'].tolist()
        
        if not user_items:
            continue
            
        # Get similar items to those in history
        user_recs = item_neighbors[item_neighbors['item_id'].isin(user_items)].copy()
        
        # Remove items the user has already interacted with
        user_recs = user_recs[~user_recs['neighbor_id'].isin(user_items)]
        
        if user_recs.empty:
            continue
            
        # Aggregate similarity scores for each recommended item
        user_recs = user_recs.groupby('neighbor_id')['sim'].sum().reset_index()
        user_recs['user_id'] = user_id
        
        # Rank recommendations
        user_recs = user_recs.sort_values('sim', ascending=False)
        user_recs['rank'] = range(1, len(user_recs) + 1)
        
        all_recs.append(user_recs)
    
    if all_recs:
        recs = pd.concat(all_recs, ignore_index=True)
        recs = recs.rename(columns={'neighbor_id': 'rec_item_id'})
        
        print("\n8. Evaluating recommendations...")
        # Join recommendations with validation data
        joined = pd.merge(
            recs, 
            val, 
            on='user_id', 
            how='left'
        )
        
        # Check if recommendations match validation items
        joined['hit'] = joined['rec_item_id'] == joined['true_item']
        
        # Calculate Recall@K
        total_val_users = val['user_id'].nunique()
        print(f"Total validation users: {total_val_users}")
        
        for k in (5, 10, 20, 50):
            # Filter to top-K recommendations
            top_k_recs = joined[joined['rank'] <= k]
            
            # Count users with hits
            users_with_hits = top_k_recs[top_k_recs['hit']]['user_id'].nunique()
            
            recall = users_with_hits / total_val_users
            print(f"Recall@{k}: {recall:.4f} ({users_with_hits} users had their validation item in top {k} recommendations)")
    else:
        print("No recommendations could be generated.")
else:
    print("Skipping recommendation evaluation (no validation data).")

print("\nItem-to-item collaborative filtering completed!")
