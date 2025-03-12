import os
import feedparser
import pandas as pd
import slack_sdk
import google.generativeai as genai
import time
import random
from datetime import datetime
import re
import urllib
import ast

# Load API keys from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load RSS Feeds from the repository (rss_feeds.txt should be in the same directory)
rss_file_path = "mnaFeeds.txt"
with open(rss_file_path, "r") as file:
    rss_feeds = [line.strip() for line in file.readlines() if line.strip()]


rss_feeds = ast.literal_eval("".join(rss_feeds))
print("✅ RSS Feeds Loaded:", rss_feeds)

# Initialize an empty list to store news articles
all_news = []

# Loop through each RSS feed URL
for rss_url in rss_feeds:
    # Parse the RSS feed
    feed = feedparser.parse(rss_url)

    # Extract relevant data
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        published = entry.published if 'published' in entry else 'Unknown Date'

        # Extract only the date (YYYY-MM-DD format) from ISO 8601
        try:
            parsed_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
        except ValueError:
            parsed_date = "Unknown Date"  # Fallback if parsing fails

        # Append to list
        all_news.append({"Date": parsed_date, "Title": title, "URL": link})

# Convert to DataFrame for better display
df = pd.DataFrame(all_news)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
today_date = datetime.today().strftime('%Y-%m-%d')
df_today = df[df['Date'].dt.strftime('%Y-%m-%d') == today_date]

# Clean HTML tags in titles
def clean_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)  # Remove HTML tags like <b>...</b>
    return clean_text.strip()

df['Title'] = df['Title'].apply(clean_html_tags)

# Extract actual URL if it's a Google redirect URL
def extract_actual_url(google_url):
    parsed_url = urllib.parse.urlparse(google_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    return query_params.get("url", [google_url])[0]  # Return the actual URL or original if not found

# Apply the function to clean the 'URL' column
df['URL'] = df['URL'].apply(extract_actual_url)

# Randomly shuffle available Gemini models
models = [
    {"name": "gemini-1.5-pro", "rpm": 60},
    {"name": "gemini-1.5-flash", "rpm": 60},
    {"name": "gemini-2.0-flash", "rpm": 15},
    {"name": "gemini-2.0-flash-lite-preview", "rpm": 30}
]
random.shuffle(models)  # Shuffle models for better load distribution

# Track API usage per model
request_count = {model["name"]: 0 for model in models}
last_request_time = {model["name"]: 0 for model in models}
global_last_request_time = 0

def enforce_rate_limit(model_name):
    """
    Ensures requests to Gemini do not exceed rate limits.
    Introduces a short delay to prevent hitting rate limits.
    """
    global global_last_request_time
    now = time.time()

    model = next((m for m in models if m["name"] == model_name), None)
    if not model:
        print(f"⚠️ Model {model_name} not found. Skipping.")
        return False

    # Introduce delay to prevent hitting rate limits
    time_since_last_request = now - last_request_time[model_name]
    min_time_between_requests = 60 / model["rpm"]

    if time_since_last_request < min_time_between_requests:
        wait_time = min_time_between_requests - time_since_last_request
        print(f"⏳ Waiting {wait_time:.2f} seconds for RPM compliance...")
        time.sleep(wait_time)

    last_request_time[model_name] = time.time()
    global_last_request_time = time.time()
    return True

def send_prompt_with_backoff(prompt, model_name):
    """
    Sends a request to Gemini AI while handling rate limits.
    Automatically switches models if failures occur.
    """
    if not enforce_rate_limit(model_name):
        return None

    attempt = 0
    MAX_RETRIES = 2  # Reduce excessive retries
    while attempt < MAX_RETRIES:
        try:
            print(f"🚀 Sending request to {model_name} (Attempt {attempt+1})")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            request_count[model_name] += 1

            if response and hasattr(response, "text"):
                return response.text.strip()

            print(f"⚠️ Model {model_name} returned empty response. Retrying...")
        except Exception as e:
            if "rate" in str(e).lower():
                sleep_time = 2 ** attempt
                print(f"⏳ Rate limit hit. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"🚨 Error with {model_name}: {e}. Trying next model...")
                break
        attempt += 1
    return None  # Return None if retries fail

def format_dataframe_for_gemini(df):
    """
    Converts DataFrame into structured text format for Gemini.
    Each news entry includes a **date**, **title (hyperlinked)**, and **source URL**.
    """
    formatted_text = "🔍 **Recent News Articles:**\n\n"
    for _, row in df.iterrows():
        formatted_text += f"- {row['Title']}, {row['URL']}\n"
    return formatted_text


def summarize_news_with_gemini(df, query):
    """
    Processes the news from DataFrame, structures it, and sends to Gemini for summarization.
    Gemini will also remove duplicates based on the **Title** before returning the final summary.
    """
    MAX_PROMPT_LENGTH = 900000

    # Convert DataFrame to text format
    news_text = format_dataframe_for_gemini(df)

    # Ensure the input is within the limit
    if len(news_text) > MAX_PROMPT_LENGTH:
        news_text = news_text[:MAX_PROMPT_LENGTH]

    prompt =  f"\n\n💡 **Query:** {query}\n\n{news_text}"

    model_index = 0
    selected_model = models[model_index]["name"]
    print(f"🚀 Processing with {selected_model}...")

    response = send_prompt_with_backoff(prompt, selected_model)

    if response:
        print("✅ Summary generated successfully!")
        return response

    print("⚠️ Failed to generate summary.")
    return None

# The query you want Gemini to summarize
query =  ("""Below news are in this format. Date: news, URL. For these news apply filter to get only those related to merger and 
acquisitions in the real estate and mortgage industry. Main focus should be in the US market.

- Please take care of the duplicate news titles from different or same source. I want only unique titles. And final news list should not exceed 20.
- ONLY THE OUTPUT FORMAT YOU CAN GIVE>News, URL
    """)
summary = summarize_news_with_gemini(df, query)

# Slack integration
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Initialize Slack WebClient
client = WebClient(token=SLACK_BOT_TOKEN)

# Define Slack Channel Name
channel_name = 'mna-news-channel'

def get_channel_id(channel_name):
    """Fetches the Slack private channel ID given the channel name."""
    try:
        response = client.conversations_list(types="private_channel")
        channels = response.get("channels", [])

        for channel in channels:
            if channel["name"] == channel_name:
                return channel["id"]

        return None

    except SlackApiError:
        return None
from datetime import datetime

def format_summary_for_slack(summary):
    """
    Formats the summary into a Slack-compatible structure with properly formatted links.

    Args:
    - summary (str): The raw summary text (comma-separated values: Date, Title, URL).

    Returns:
    - str: A properly formatted Slack message.
    """
    formatted_summary = "*🏡 Real Estate Market M&A Updates*\n\n"
    lines = summary.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Split by commas to get Date, Title, and URL
        parts = line.split(", ")
        if len(parts) < 3:
            continue  # Skip invalid lines

        date, title, url = parts[0].strip(), parts[1].strip(), parts[2].strip()

        # Format for Slack using `<URL|Title>` format
        formatted_summary += f"- <{url}|{title}>\n"

    return formatted_summary.strip()

def send_message_to_slack(channel_id, summary_text, slack_token):
    """
    Sends the preformatted summary message to Slack.

    Args:
    - channel_id (str): The Slack channel ID.
    - summary_text (str): The already formatted summary text.
    - slack_token (str): Slack Bot Token.
    """
    if not channel_id:
        print("❌ Cannot send message: Channel ID not found.")
        return

    client = WebClient(token=slack_token)

    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=summary_text,  # Directly sending the preformatted summary
            mrkdwn=True  # Ensures Slack processes Markdown formatting correctly
        )
        print("✅ Message sent successfully to Slack!")
    except SlackApiError as e:
        print(f"❌ Error sending message: {e.response['error']}")

# Format the summary and send to Slack
formatted_summary = format_summary_for_slack(summary)
channel_id = get_channel_id(channel_name)
print(summary)
print("---------------------------------")
print(formatted_summary)
if formatted_summary and channel_id:
    send_message_to_slack(channel_id, formatted_summary, SLACK_BOT_TOKEN)
