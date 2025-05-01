import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable

def precision_at_k(recommended_items, relevant_items, k):
    """Calculate precision@k."""
    recommended_k = recommended_items[:k]
    relevant_recommended = set(recommended_k) & set(relevant_items)
    return len(relevant_recommended) / k if k > 0 else 0

def recall_at_k(recommended_items, relevant_items, k):
    """Calculate recall@k."""
    recommended_k = recommended_items[:k]
    relevant_recommended = set(recommended_k) & set(relevant_items)
    return len(relevant_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0

def f1_at_k(recommended_items, relevant_items, k):
    """Calculate F1@k."""
    p = precision_at_k(recommended_items, relevant_items, k)
    r = recall_at_k(recommended_items, relevant_items, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    """Calculate NDCG@k."""
    import numpy as np
    
    # Create a relevance array (1 if item is relevant, 0 otherwise)
    relevance = np.array([1 if item in relevant_items else 0 for item in recommended_items[:k]])
    
    # Calculate DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    
    # Calculate ideal DCG (IDCG)
    ideal_relevance = np.ones(min(len(relevant_items), k))
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    return dcg / idcg if idcg > 0 else 0

def diversity(recommended_items, item_features):
    """Calculate diversity as average pairwise dissimilarity."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    # Get feature vectors for recommended items
    feature_vectors = [item_features[item] for item in recommended_items]
    
    # Calculate pairwise distances
    if len(feature_vectors) > 1:
        distances = pdist(feature_vectors, 'cosine')
        return np.mean(distances)
    return 0

def catalog_coverage(all_recommendations, catalog_size):
    """
    Calculate catalog coverage - the percentage of items in the catalog that are recommended to at least one user.
    
    Parameters:
    -----------
    all_recommendations : list of lists
        List containing recommendation lists for each user
    catalog_size : int
        Total number of items in the catalog
        
    Returns:
    --------
    float
        Percentage of catalog covered by recommendations
    """
    unique_recommended_items = set()
    for user_recommendations in all_recommendations:
        unique_recommended_items.update(user_recommendations)
    
    return len(unique_recommended_items) / catalog_size if catalog_size > 0 else 0

def novelty(recommended_items, item_popularity, total_interactions):
    """
    Calculate novelty as the average inverse popularity of recommended items.
    Higher values indicate more novel (less popular) recommendations.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    item_popularity : dict
        Dictionary mapping item IDs to their interaction counts
    total_interactions : int
        Total number of interactions in the system
        
    Returns:
    --------
    float
        Novelty score (higher means more novel recommendations)
    """
    import numpy as np
    
    # Calculate relative popularity (probability of being interacted with)
    rel_popularity = {item: count/total_interactions for item, count in item_popularity.items()}
    
    # Calculate self-information (inverse of popularity)
    novelty_scores = [-np.log2(rel_popularity.get(item, 1/total_interactions)) for item in recommended_items]
    
    return np.mean(novelty_scores) if novelty_scores else 0

def serendipity(recommended_items, expected_items, relevant_items):
    """
    Calculate serendipity as the proportion of unexpected and relevant recommendations.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    expected_items : list
        List of items that would be expected (e.g., from a popularity model)
    relevant_items : list
        List of items that are actually relevant to the user
        
    Returns:
    --------
    float
        Serendipity score
    """
    unexpected_items = [item for item in recommended_items if item not in expected_items]
    unexpected_relevant = [item for item in unexpected_items if item in relevant_items]
    
    return len(unexpected_relevant) / len(recommended_items) if recommended_items else 0

def filter_available_items(candidate_items, item_timestamps, cutoff_time):
    """
    Filter items to include only those available before the cutoff time.
    
    Parameters:
    -----------
    candidate_items : list
        List of candidate item IDs
    item_timestamps : dict
        Dictionary mapping item IDs to their creation/publication timestamps
    cutoff_time : datetime
        Timestamp to use as cutoff (only items published before this time are valid)
        
    Returns:
    --------
    list
        Filtered list of item IDs that were available before the cutoff time
    """
    return [item for item in candidate_items 
            if item in item_timestamps and item_timestamps[item] <= cutoff_time]

def evaluate_for_user(user_id, 
                     recommended_items, 
                     relevant_items, 
                     k_values=[5, 10, 20],
                     item_timestamps=None,
                     cutoff_time=None,
                     item_features=None, 
                     expected_items=None,
                     item_popularity=None, 
                     total_interactions=None):
    """
    Evaluate recommendations for a single user with time-based filtering.
    
    Parameters:
    -----------
    user_id : any
        User identifier
    recommended_items : list
        List of recommended item IDs for the user
    relevant_items : list
        List of items that are relevant to the user (ground truth)
    k_values : list
        List of k values to evaluate metrics at
    item_timestamps : dict, optional
        Dictionary mapping item IDs to their creation/publication timestamps
    cutoff_time : datetime, optional
        Timestamp to use as cutoff (only items published before this are valid)
    item_features : dict, optional
        Dictionary mapping item IDs to feature vectors (for diversity)
    expected_items : list, optional
        List of expected items (for serendipity)
    item_popularity : dict, optional
        Dictionary mapping item IDs to their popularity (for novelty)
    total_interactions : int, optional
        Total number of interactions (for novelty)
        
    Returns:
    --------
    dict
        Dictionary of metrics at different k values for this user
    """
    # Filter recommendations if time constraints are provided
    filtered_recommendations = recommended_items
    if item_timestamps is not None and cutoff_time is not None:
        filtered_recommendations = filter_available_items(
            recommended_items, item_timestamps, cutoff_time
        )
    
    # Calculate metrics for each k
    results = {}
    for k in k_values:
        if k <= len(filtered_recommendations):
            results[f'precision@{k}'] = precision_at_k(filtered_recommendations, relevant_items, k)
            results[f'recall@{k}'] = recall_at_k(filtered_recommendations, relevant_items, k)
            results[f'f1@{k}'] = f1_at_k(filtered_recommendations, relevant_items, k)
            results[f'ndcg@{k}'] = ndcg_at_k(filtered_recommendations, relevant_items, k)
            
            # Calculate diversity if item features are provided
            if item_features is not None:
                results[f'diversity@{k}'] = diversity(filtered_recommendations[:k], item_features)
            
            # Calculate serendipity if expected items are provided
            if expected_items is not None:
                filtered_expected = expected_items
                if item_timestamps is not None and cutoff_time is not None:
                    filtered_expected = filter_available_items(
                        expected_items, item_timestamps, cutoff_time
                    )
                results[f'serendipity@{k}'] = serendipity(
                    filtered_recommendations[:k], filtered_expected[:k], relevant_items
                )
            
            # Calculate novelty if item popularity is provided
            if item_popularity is not None and total_interactions is not None:
                results[f'novelty@{k}'] = novelty(
                    filtered_recommendations[:k], item_popularity, total_interactions
                )
        else:
            # Not enough recommendations after filtering
            for metric in ['precision', 'recall', 'f1', 'ndcg']:
                results[f'{metric}@{k}'] = 0
            if item_features is not None:
                results[f'diversity@{k}'] = 0
            if expected_items is not None:
                results[f'serendipity@{k}'] = 0
            if item_popularity is not None:
                results[f'novelty@{k}'] = 0
    
    return results

def evaluate_recommendations_df(recommendations_df, 
                              test_df, 
                              articles_df,
                              k_values=[5, 10, 20], 
                              item_features=None,
                              popularity_recommendations=None,
                              item_popularity=None,
                              total_interactions=None):
    """
    Evaluate recommendations using a DataFrame-based approach with time-based filtering.
    
    Parameters:
    -----------
    recommendations_df : DataFrame
        DataFrame with columns 'article_id' and optionally 'score'
    test_df : DataFrame
        Test data with columns 'user_id', 'click_article_id', and 'click_timestamp'
    articles_df : DataFrame
        Article metadata with columns 'article_id' and 'created_at_ts'
    k_values : list
        List of k values to evaluate metrics at
    item_features : dict, optional
        Dictionary mapping item IDs to feature vectors (for diversity)
    popularity_recommendations : DataFrame, optional
        DataFrame of popularity-based recommendations (for serendipity)
    item_popularity : dict, optional
        Dictionary mapping item IDs to their popularity (for novelty)
    total_interactions : int, optional
        Total number of interactions (for novelty)
        
    Returns:
    --------
    dict
        Dictionary of average metrics at different k values
    """
    # Extract article publication timestamps
    pub_dates = articles_df.set_index("article_id")["created_at_ts"].to_dict()
    
    # Get list of recommended article IDs
    if 'score' in recommendations_df.columns:
        # Sort by score if available
        recommendations_df = recommendations_df.sort_values('score', ascending=False)
    art_ids = recommendations_df["article_id"].tolist()
    
    # Get popularity recommendations if available
    pop_ids = None
    if popularity_recommendations is not None:
        if 'score' in popularity_recommendations.columns:
            popularity_recommendations = popularity_recommendations.sort_values('score', ascending=False)
        pop_ids = popularity_recommendations["article_id"].tolist()
    
    # Group test data by user
    user_metrics = {}
    for user_id, user_df in test_df.groupby("user_id"):
        # Get user's cutoff time (last interaction time)
        cutoff = user_df["click_timestamp"].max()
        
        # Get relevant items (actual clicks)
        relevant_items = user_df["click_article_id"].tolist()
        
        # Evaluate for this user
        user_metrics[user_id] = evaluate_for_user(
            user_id=user_id,
            recommended_items=art_ids,
            relevant_items=relevant_items,
            k_values=k_values,
            item_timestamps=pub_dates,
            cutoff_time=cutoff,
            item_features=item_features,
            expected_items=pop_ids,
            item_popularity=item_popularity,
            total_interactions=total_interactions
        )
    
    # Aggregate metrics across users
    results = {}
    metrics_to_aggregate = set()
    for user_data in user_metrics.values():
        metrics_to_aggregate.update(user_data.keys())
    
    for metric in metrics_to_aggregate:
        values = [user_data.get(metric, 0) for user_data in user_metrics.values()]
        results[metric] = sum(values) / len(values) if values else 0
    
    return results

def evaluate_all_models(model_recommendations, 
                       test_df, 
                       articles_df,
                       k_values=[5, 10, 20], 
                       item_features=None,
                       popularity_model=None,
                       item_popularity=None,
                       total_interactions=None):
    """
    Evaluate multiple recommendation models and compare their performance.
    
    Parameters:
    -----------
    model_recommendations : dict
        Dictionary mapping model names to their recommendation DataFrames
    test_df : DataFrame
        Test data with columns 'user_id', 'click_article_id', and 'click_timestamp'
    articles_df : DataFrame
        Article metadata with columns 'article_id' and 'created_at_ts'
    k_values : list
        List of k values to evaluate metrics at
    item_features : dict, optional
        Dictionary mapping item IDs to feature vectors (for diversity)
    popularity_model : str, optional
        Name of the popularity model in model_recommendations (for serendipity)
    item_popularity : dict, optional
        Dictionary mapping item IDs to their popularity (for novelty)
    total_interactions : int, optional
        Total number of interactions (for novelty)
        
    Returns:
    --------
    dict
        Dictionary mapping model names to their evaluation metrics
    """
    results = {}
    
    # Get popularity recommendations if specified
    pop_recs = None
    if popularity_model is not None and popularity_model in model_recommendations:
        pop_recs = model_recommendations[popularity_model]
    
    # Evaluate each model
    for model_name, recs_df in model_recommendations.items():
        print(f"Evaluating {model_name}...")
        model_metrics = evaluate_recommendations_df(
            recommendations_df=recs_df,
            test_df=test_df,
            articles_df=articles_df,
            k_values=k_values,
            item_features=item_features,
            popularity_recommendations=pop_recs if model_name != popularity_model else None,
            item_popularity=item_popularity,
            total_interactions=total_interactions
        )
        results[model_name] = model_metrics
    
    return results

def format_evaluation_results(results, metrics_to_show=None, k_values=None):
    """
    Format evaluation results into a DataFrame for easy comparison.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping model names to their evaluation metrics
    metrics_to_show : list, optional
        List of metrics to include in the output
    k_values : list, optional
        List of k values to include in the output
        
    Returns:
    --------
    DataFrame
        DataFrame with models as rows and metrics as columns
    """
    import pandas as pd
    
    # Determine which metrics and k values to show
    all_metrics = set()
    all_k_values = set()
    for model_metrics in results.values():
        for metric_name in model_metrics:
            metric_parts = metric_name.split('@')
            if len(metric_parts) == 2:
                metric, k = metric_parts
                all_metrics.add(metric)
                all_k_values.add(int(k))
    
    # Filter metrics and k values if specified
    metrics = sorted(list(metrics_to_show or all_metrics))
    k_vals = sorted(list(k_values or all_k_values))
    
    # Create DataFrame
    data = []
    for model_name, model_metrics in results.items():
        row = {'Model': model_name}
        for metric in metrics:
            for k in k_vals:
                key = f'{metric}@{k}'
                row[key] = model_metrics.get(key, float('nan'))
        data.append(row)
    
    return pd.DataFrame(data).set_index('Model')

def evaluate_recommendations(recommended_items, relevant_items, k_values=[5, 10, 20], 
                            item_features=None, expected_items=None, 
                            item_popularity=None, total_interactions=None):
    """
    Evaluate recommendations using multiple metrics at different k values.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of items that are relevant to the user
    k_values : list
        List of k values to evaluate metrics at
    item_features : dict, optional
        Dictionary mapping item IDs to feature vectors (for diversity)
    expected_items : list, optional
        List of expected items (for serendipity)
    item_popularity : dict, optional
        Dictionary mapping item IDs to their popularity (for novelty)
    total_interactions : int, optional
        Total number of interactions (for novelty)
        
    Returns:
    --------
    dict
        Dictionary of metrics at different k values
    """
    results = {}
    
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(recommended_items, relevant_items, k)
        results[f'recall@{k}'] = recall_at_k(recommended_items, relevant_items, k)
        results[f'f1@{k}'] = f1_at_k(recommended_items, relevant_items, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommended_items, relevant_items, k)
    
    # Calculate diversity if item features are provided
    if item_features is not None:
        for k in k_values:
            results[f'diversity@{k}'] = diversity(recommended_items[:k], item_features)
    
    # Calculate serendipity if expected items are provided
    if expected_items is not None and relevant_items is not None:
        for k in k_values:
            results[f'serendipity@{k}'] = serendipity(
                recommended_items[:k], expected_items[:k], relevant_items
            )
    
    # Calculate novelty if item popularity is provided
    if item_popularity is not None and total_interactions is not None:
        for k in k_values:
            results[f'novelty@{k}'] = novelty(
                recommended_items[:k], item_popularity, total_interactions
            )
    
    return results