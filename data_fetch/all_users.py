import requests
import pandas as pd

# Base URL for the API
base_url = "https://api.socialverseapp.com/users/get_all?page=1&page_size=1000"
headers={"Flic-Token": "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"}
# Initialize variables
all_users = []
current_page = 1  # Start with page 1

while True:
    # Make the GET request for the current page
    params = {"page": current_page}
    response = requests.get(base_url,headers=headers,params=params)

    # Check if the response is valid
    if response.status_code == 200:
        data = response.json()

        # Check if "posts" exist and add them to the list
        if "users" in data and data["users"]:
            all_users.extend(data["users"])
            print(f"Fetched {len(data['users'])} posts from page {current_page}.")
        else:
            print(f"No more users found. Stopping at page {current_page}.")
            break

        # Move to the next page
        current_page += 1
    else:
        print(f"Error: Unable to fetch page {current_page}, Status Code: {response.status_code}")
        break



# Convert the combined data into a DataFrame
if all_users:
    df = pd.DataFrame(all_users)
    print(f"\nDataFrame created with {len(df)} records and {len(df.columns)} columns.")

    # Save to a CSV file for local storage
    df.to_csv("all_users.csv", index=False)
    print("Data saved to 'all_users.csv'.")

    # Optional: Display the first few rows of the DataFrame
    print("\nPreview of the DataFrame:")
    print(df.head())
else:
    print("No data fetched.")

