# Recommendation System Project

This project implements a hybrid recommendation system combining collaborative filtering (SVD-based) and content-based filtering (TF-IDF-based) to recommend posts to users based on their interactions.

## Project Overview

The system uses data from a user-item interaction matrix and content features (such as post titles) to generate personalized recommendations for users. The system leverages two main approaches:

1. **Collaborative Filtering**: Uses Singular Value Decomposition (SVD) to learn user preferences and make predictions.
2. **Content-Based Filtering**: Uses TF-IDF to calculate similarity between posts based on their content (titles).

The recommendations are generated by combining both collaborative and content-based methods using a weighted approach.

## Features

- Hybrid recommendation system combining collaborative and content-based filtering
- Generates personalized post recommendations for users
- Evaluates the recommendations using precision, recall, MRR, and NDCG metrics
- API endpoints for providing recommendations based on user input

## Prerequisites

Make sure you have Python 3.x installed and the following libraries:

- pandas
- numpy
- scikit-learn
- flask

You can install them using `pip`:

```bash
pip install -r requirements.txt.
```

## Data
The project uses a dataset consisting of user interactions with posts, which includes:

User ID: Identifies the user

Post ID: Identifies the post

Interaction Type: Defines the type of interaction (liked, viewed, rated)

Title: Title of the post

Category ID: Category of the post

video link:links to the video

Mood: User mood 

The dataset is expected to be in CSV format (filtered_combined_interactions2.csv).

## API Endpoints
The following API endpoints are available for generating post recommendations:
```
GET /feed?username=your_username: Get recommendations for a specific user.
GET /feed?username=your_username&category_id=category_id_user_want_to_see: Get recommendations filtered by category.
GET /feed?username=your_username&category_id=category_id_user_want_to_see&mood=user_current_mood: Get recommendations filtered by category and mood.
Example API Request:
bash
Copy code
http://localhost:5000/feed?username=your_username&category_id=1&mood=happy
```

## Evaluation Metrics
The system uses the following evaluation metrics:

Precision@K: Measures the accuracy of recommendations.

Recall@K: Measures the completeness of recommendations.

Mean Reciprocal Rank (MRR): Measures the rank of the first relevant recommendation.

Normalized Discounted Cumulative Gain (NDCG@K): Measures the ranking quality of recommendations.

## Precision Improvement
While the current precision is 55%, improvements can be made to increase this metric by:

Fine-tuning the alpha parameter: Adjusting the weight of the hybrid recommendation system (alpha controls the contribution of collaborative vs. content-based filtering).

Enhancing the content features: Using more detailed post summaries, keywords, or metadata to improve content-based recommendations.

Optimizing the number of SVD components: Experimenting with a different number of components in SVD could result in better collaborative filtering performance.

Incorporating additional features: Adding more user and post features such as tags, categories, and interactions to refine the recommendations further.

By addressing these areas, the precision of the recommendation system can be improved, leading to more accurate and relevant recommendations for users.


