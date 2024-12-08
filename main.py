from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Use relative path
current_dir = os.path.dirname(os.path.realpath(__file__))  # Gets the directory of the script
file_path = os.path.join(current_dir, 'data_fetch', 'filtered_combined_interactions2.csv')

# Load the dataset using the relative path
full_data = pd.read_csv(file_path)

# Step 1: Create the User-Item Interaction Matrix
full_data['interaction_weight'] = full_data['interaction_type'].map({'liked': 1, 'viewed': 1, 'rated': 2})
interaction_matrix = full_data.pivot_table(index='user_id', columns='post_id', values='interaction_weight', fill_value=0)

# Step 2: Apply SVD for Collaborative Filtering
svd = TruncatedSVD(n_components=50)
svd_matrix = svd.fit_transform(interaction_matrix)

# Step 3: Compute Content-Based Similarity (TF-IDF)
full_data['title'] = full_data['title'].fillna('')  # Replace NaN with empty string
posts_in_interactions = interaction_matrix.columns
filtered_data = full_data[full_data['post_id'].isin(posts_in_interactions)]

tfidf = TfidfVectorizer(stop_words='english')
posts_tfidf = tfidf.fit_transform(filtered_data['title'])
post_similarity = cosine_similarity(posts_tfidf)

# Align the post_similarity matrix to match the interaction_matrix columns
post_ids = list(interaction_matrix.columns)
post_id_to_index = {post_id: idx for idx, post_id in enumerate(filtered_data['post_id'])}
aligned_similarity = np.zeros((len(post_ids), len(post_ids)))
for i, post_id_1 in enumerate(post_ids):
    for j, post_id_2 in enumerate(post_ids):
        idx1 = post_id_to_index[post_id_1]
        idx2 = post_id_to_index[post_id_2]
        aligned_similarity[i, j] = post_similarity[idx1, idx2]

def hybrid_recommendation(user_id, interaction_matrix, svd_matrix, aligned_similarity, k=10, alpha=0.5):
    try:
        user_index = interaction_matrix.index.get_loc(user_id)
        user_svd = svd_matrix[user_index, :]

        predicted_ratings = np.dot(user_svd, svd.components_)

        user_posts = np.where(interaction_matrix.loc[user_id] > 0)[0]
        content_similarities = np.mean(aligned_similarity[user_posts], axis=0)

        combined_scores = alpha * predicted_ratings + (1 - alpha) * content_similarities
        recommended_post_indices = np.argsort(combined_scores)[-k:][::-1]
        recommended_posts = interaction_matrix.columns[recommended_post_indices]

        return recommended_posts
    except Exception as e:
        print(f"Error in hybrid recommendation: {e}")
        return []

@app.route('/feed', methods=['GET'])
def get_feed():
    try:
        username = request.args.get('username')
        category_id = request.args.get('category_id', type=int)
        mood = request.args.get('Mood')

        # Convert the username to user_id
        user_id = username_to_user_id(username)

        if user_id is None:
            return jsonify({'error': 'User not found', 'username': username}), 404

        # Get recommendations based on parameters
        recommended_posts = hybrid_recommendation(user_id, interaction_matrix, svd_matrix, aligned_similarity, k=10)

        if category_id is not None and mood is not None:
            recommended_posts = filter_posts_by_category_and_mood(recommended_posts, category_id, mood)
        elif category_id is not None:
            recommended_posts = filter_posts_by_category(recommended_posts, category_id)

        # Remove duplicates and ensure unique post_id to video_link mapping
        recommended_posts = list(set(recommended_posts))

        # Check if recommended post_ids exist in the full_data
        missing_post_ids = [post for post in recommended_posts if post not in full_data['post_id'].values]
        if missing_post_ids:
            print(f"Missing post IDs: {missing_post_ids}")
            # Optionally, handle missing post_ids

        # Remove duplicate post_ids, keeping the first occurrence (or you can use another strategy)
        post_data_unique = full_data.drop_duplicates(subset=['post_id'])

        # Retrieve the post data for recommended post_ids
        post_data_filtered = post_data_unique[post_data_unique['post_id'].isin(recommended_posts)]

        # Ensure video links are aligned with post_ids
        video_links = post_data_filtered.set_index('post_id')['video_link'].reindex(recommended_posts).fillna(
            'No video available').tolist()

        return jsonify({
            'username': username,
            'recommended_posts': recommended_posts,
            'video_links': video_links
        })
    except Exception as e:
        logging.error(f"Error in /feed endpoint: {e}")
        return jsonify({'error': str(e), 'username': username}), 500



def username_to_user_id(username):
    try:
        user = full_data[full_data['username'] == username]
        return user['user_id'].values[0] if not user.empty else None
    except Exception as e:
        print(f"Error converting username to user_id: {e}")
        return None

def filter_posts_by_category_and_mood(recommended_posts, category_id, mood):
    return [post for post in recommended_posts if matches_category_and_mood(post, category_id, mood)]

def filter_posts_by_category(recommended_posts, category_id):
    return [post for post in recommended_posts if matches_category(post, category_id)]

def matches_category_and_mood(post, category_id, mood):
    try:
        post_data = full_data[full_data['post_id'] == post]
        return (not post_data.empty and
                post_data.iloc[0]['category_id'] == category_id and
                post_data.iloc[0]['Mood'] == mood)
    except Exception as e:
        print(f"Error in matches_category_and_mood: {e}")
        return False

def matches_category(post, category_id):
    try:
        post_data = full_data[full_data['post_id'] == post]
        return not post_data.empty and post_data.iloc[0]['category_id'] == category_id
    except Exception as e:
        print(f"Error in matches_category: {e}")
        return False

if __name__ == '__main__':
    app.run(port=5000, debug=True)