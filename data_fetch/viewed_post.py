import requests
import pandas as pd

# Base URL for the API
base_url = "https://api.socialverseapp.com/posts/view?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"


# Initialize variables
all_posts = []
current_page = 1  # Start with page 1

while True:
    # Make the GET request for the current page
    params = {"page": current_page}
    response = requests.get(base_url, params=params)

    # Check if the response is valid
    if response.status_code == 200:
        data = response.json()

        # Check if "posts" exist and add them to the list
        if "posts" in data and data["posts"]:
            all_posts.extend(data["posts"])
            print(f"Fetched {len(data['posts'])} posts from page {current_page}.")
        else:
            print(f"No more posts found. Stopping at page {current_page}.")
            break

        # Move to the next page
        current_page += 1
    else:
        print(f"Error: Unable to fetch page {current_page}, Status Code: {response.status_code}")
        break

# Convert the combined data into a DataFrame
if all_posts:
    df = pd.DataFrame(all_posts)
    print(f"\nDataFrame created with {len(df)} records and {len(df.columns)} columns.")

    # Save to a CSV file for local storage
    df.to_csv("viewed_posts.csv", index=False)
    print("Data saved to 'viewed_posts.csv'.")
else:
    print("No posts fetched.")

# Optional: Display the first few rows of the DataFrame
print("\nPreview of the DataFrame:")
print(df.head())
