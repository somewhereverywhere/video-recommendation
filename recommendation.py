import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the merged dataset
full_data = pd.read_csv(r'C:\Users\hanan\PycharmProjects\pythonProject\data_fetch\filtered_combined_interactions2.csv')

# Step 1: Create the User-Item Interaction Matrix
full_data['interaction_weight'] = full_data['interaction_type'].map({'liked': 1, 'viewed': 0.5, 'rated': 2})
interaction_matrix = full_data.pivot_table(index='user_id', columns='post_id', values='interaction_weight',
                                           fill_value=0)

# Step 2: Apply SVD for Collaborative Filtering
svd = TruncatedSVD(n_components=50)  # Experiment with the number of components
svd_matrix = svd.fit_transform(interaction_matrix)

# Step 3: Compute the Content-Based Similarity (TF-IDF)
full_data['title'] = full_data['title'].fillna('')  # Replace NaN with empty string

# Combine title and post_summary into one text field

# Filter data to ensure only posts present in the interaction_matrix are used
posts_in_interactions = interaction_matrix.columns
filtered_data = full_data[full_data['post_id'].isin(posts_in_interactions)]

# Apply TF-IDF only to filtered posts
tfidf = TfidfVectorizer(stop_words='english')
posts_tfidf = tfidf.fit_transform(filtered_data['title'])

# Compute cosine similarity between posts
post_similarity = cosine_similarity(posts_tfidf)

# Align the post_similarity matrix to match the interaction_matrix columns
post_ids = list(interaction_matrix.columns)  # Post IDs in the interaction matrix
post_id_to_index = {post_id: idx for idx, post_id in enumerate(filtered_data['post_id'])}

# Create a filtered post_similarity matrix aligned with the interaction_matrix
aligned_similarity = np.zeros((len(post_ids), len(post_ids)))
for i, post_id_1 in enumerate(post_ids):
    for j, post_id_2 in enumerate(post_ids):
        idx1 = post_id_to_index[post_id_1]
        idx2 = post_id_to_index[post_id_2]
        aligned_similarity[i, j] = post_similarity[idx1, idx2]


# Step 4: Generate Recommendations using Hybrid Approach
def hybrid_recommendation(user_id, interaction_matrix, svd_matrix, aligned_similarity, k=10, alpha=0.5):
    # Collaborative Filtering (SVD-based)
    user_index = interaction_matrix.index.get_loc(user_id)
    user_svd = svd_matrix[user_index, :]

    # Compute the predicted ratings for posts using SVD
    predicted_ratings = np.dot(user_svd, svd.components_)

    # Content-Based Filtering (TF-IDF-based)
    user_posts = np.where(interaction_matrix.loc[user_id] > 0)[0]  # Get indices of user's interacted posts
    content_similarities = np.mean(aligned_similarity[user_posts], axis=0)

    # Combine both collaborative and content-based filtering results
    combined_scores = alpha * predicted_ratings + (1 - alpha) * content_similarities

    # Get the top-k recommended posts
    recommended_post_indices = np.argsort(combined_scores)[-k:][::-1]
    recommended_posts = interaction_matrix.columns[recommended_post_indices]

    return recommended_posts


