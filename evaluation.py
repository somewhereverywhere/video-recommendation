from sklearn.metrics import precision_score, recall_score
import numpy as np
from recommendation import hybrid_recommendation,interaction_matrix, svd_matrix, aligned_similarity


# Step 1: Define a function to compute evaluation metrics
def evaluate_recommendations(interaction_matrix, recommendations, k=5):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for user_id in interaction_matrix.index:
        # Get the ground truth (actual interactions)
        actual_interactions = interaction_matrix.loc[user_id]
        relevant_items = set(actual_interactions[actual_interactions > 0].index)

        # Get top-K recommended items
        recommended_items = recommendations[user_id][:k]
        recommended_set = set(recommended_items)

        # Compute Precision@K
        true_positives = len(relevant_items & recommended_set)
        precision = true_positives / k
        precision_scores.append(precision)

        # Compute Recall@K
        recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0
        recall_scores.append(recall)

        # Compute Mean Reciprocal Rank (MRR)
        rank = 0
        for idx, item in enumerate(recommended_items):
            if item in relevant_items:
                rank = idx + 1
                break
        mrr = 1 / rank if rank > 0 else 0
        mrr_scores.append(mrr)

        # Compute NDCG@K
        dcg = 0
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Compute average metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_mrr = np.mean(mrr_scores)
    avg_ndcg = np.mean(ndcg_scores)

    return {
        'Precision@K': avg_precision,
        'Recall@K': avg_recall,
        'MRR': avg_mrr,
        'NDCG@K': avg_ndcg
    }


# Step 2: Generate recommendations for all users
def generate_recommendations_for_all(interaction_matrix, svd_matrix, aligned_similarity, k=10):
    recommendations = {}
    for user_id in interaction_matrix.index:
        recommendations[user_id] = hybrid_recommendation(
            user_id, interaction_matrix, svd_matrix, aligned_similarity, k=k
        )
    return recommendations


# Generate recommendations
recommendations = generate_recommendations_for_all(interaction_matrix, svd_matrix, aligned_similarity, k=5)

# Step 3: Evaluate the recommendations
metrics = evaluate_recommendations(interaction_matrix, recommendations, k=5)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
