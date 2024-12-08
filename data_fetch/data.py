import pandas as pd
import ast  # To safely evaluate stringified dictionaries


# Load the datasets
liked_df = pd.read_csv("liked_posts.csv")
inspired_df = pd.read_csv("inspired_posts.csv")
viewed_df = pd.read_csv("viewed_posts.csv")
rated_df = pd.read_csv("rated_posts.csv")
posts_df = pd.read_csv("all_posts.csv")
users_df = pd.read_csv("all_users.csv")

# Step 1: Rename columns for consistency
posts_df.rename(columns={'id': 'post_id', 'username': 'author_username'}, inplace=True)
users_df.rename(columns={'id': 'user_id'}, inplace=True)

# Step 2: Add interaction type to interaction datasets
liked_df['interaction_type'] = 'liked'
inspired_df['interaction_type'] = 'inspired'
viewed_df['interaction_type'] = 'viewed'
rated_df['interaction_type'] = 'rated'

# Combine all interaction data
interaction_frames = [liked_df, inspired_df, viewed_df, rated_df]
interactions = pd.concat(interaction_frames, ignore_index=True)

# Step 3: Merge interactions with users metadata solely on user_id
interactions_with_users = pd.merge(interactions, users_df, how='left', on='user_id')

# Step 4: Merge with posts metadata using post_id for post-related details
full_data = pd.merge(interactions_with_users, posts_df, how='left', on='post_id')


def extract_category_info(category):
    try:
        # If it's a string representation of a dictionary, convert it
        if isinstance(category, str):
            category = ast.literal_eval(category)

        # Check if it's a dictionary and extract values
        if isinstance(category, dict):
            category_id = category.get('id')
            category_name = category.get('description')
            return pd.Series([category_id, category_name])
        else:
            return pd.Series([None, None])  # Return None if not a dictionary
    except Exception as e:
        print(f"Error processing category: {e}")
        return pd.Series([None, None])


# Apply the function to create new columns
full_data[['category_id', 'category_description']] = full_data['category'].apply(extract_category_info)

# Step 5: Select only the relevant columns
full_data = full_data[[
    "user_id", "username", "post_id", "title", "view_count",
    "average_rating", "upvote_count", "interaction_type","video_link","category_id","category_description"
]]

# Step 6: Handle duplicates and missing values
full_data.drop_duplicates(inplace=True)
full_data.fillna({'average_rating': 0, 'view_count': 0, 'upvote_count': 0}, inplace=True)

# Ensure no user_ids are lost
full_data = full_data.dropna(subset=['user_id'])  # Drop rows with missing user_id

# Step 7: Save the final DataFrame to a CSV
full_data.to_csv("filtered_combined_interactions2.csv", index=False)
print("Filtered merged dataset saved to 'filtered_combined_interactions2.csv'.")

# Step 8: Inspect the dataset
print(f"Total records in filtered dataset: {len(full_data)}")
print(f"Columns in filtered dataset: {full_data.columns.tolist()}")
print("\nPreview of the filtered dataset:")
print(full_data.head())
