import requests
import json
import pandas as pd

# Base URL for the API
base_url = "https://api.socialverseapp.com/posts/like?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"

# Initialize variables
all_posts = []
total_pages = 2

for page in range(1, total_pages + 1):
    # Make the GET request for each page
    params = {"page": page}
    response = requests.get(base_url, params=params)

    # Check if the response is valid
    if response.status_code == 200:
        data = response.json()
        if "posts" in data and data["posts"]:
            all_posts.extend(data["posts"])
            print(f"Fetched {len(data['posts'])} posts from page {page}.")
        else:
            print(f"No posts found on page {page}.")
    else:
        print(f"Error: Unable to fetch page {page}, Status Code: {response.status_code}")

# Convert the fetched data into a DataFrame
if all_posts:
    df = pd.DataFrame(all_posts)
    print(f"\nDataFrame created with {len(df)} records and {len(df.columns)} columns.")

    # Save to a CSV file for local storage
    df.to_csv("liked_posts.csv", index=False)
    print("Data saved to 'liked_posts.csv'.")
else:
    print("No posts fetched.")

# Optional: Display the first few rows of the DataFrame
print("\nPreview of the DataFrame:")
print(df.head())
