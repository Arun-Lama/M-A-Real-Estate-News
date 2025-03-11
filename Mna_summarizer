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

# Load API keys from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load RSS Feeds from the repository (rss_feeds.txt should be in the same directory)
rss_file_path = "rss_feeds.txt"
with open(rss_file_path, "r") as file:
    rss_feeds = [line.strip() for line in file.readlines() if line.strip()]

print("✅ RSS Feeds Loaded:", rss_feeds)
rss_feeds = [url.strip("[]',") for url in rss_feeds]  # Clean unwanted characters

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
    formatted_text = "🔍 **Recent News:**\n\n"
    for _, row in df.iterrows():
        formatted_text += f"- **{row['Date']}**: [{row['Title']}]({row['URL']})\n"
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
query =  ("""From the following news return me only those related to:
- **Home buying platforms**
- **New real estate product launches**
- **First-time home buyers**
- **Market expansion in the real estate sector**
- **Major strategic shifts by real estate companies**
- **Blockchain and tokenization in real estate**
- **Trends in real estate technology (PropTech) adoption**
- **Investment and funding rounds in real estate startups and companies**
- **Regulatory changes and government policies impacting the real estate market**
- **Real estate mergers, acquisitions, and partnerships**
- **Innovations in real estate financing (e.g., crowdfunding, real estate REITs)**
- **Consumer behavior shifts and preferences in the real estate market**

📌 **Important Instructions for filter:**
- I don't want more than 20 news. So give me only top 20 news that are very important to know for reAlpha Tech Corp. 
  Don't give me the news that is generic and informational blog articles. I need news about important strategic moves of competitors. 
  Rather than additional texts, I would simply love the news title with hyperlink containing URL.
    """)
summary = summarize_news_with_gemini(df, query)

# Slack integration
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Initialize Slack WebClient
client = WebClient(token=SLACK_BOT_TOKEN)

# Define Slack Channel Name
channel_name = 'news-channel-2'

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
    Formats the summary for Slack:
    - Removes unnecessary asterisks around the date
    - Removes calendar emoji
    - Includes the date and title (hyperlinked) without repeating text or extra 'Link'
    """
    formatted_summary = "Here’s the latest update on the real estate sector:\n\n"
    
    # Process each line from the summary
    for line in summary.split("\n"):
        if "[" in line and "](" in line:  # Identifying Slack hyperlink formatting
            # Find the title and URL in the markdown format
            title_start = line.find("[") + 1
            title_end = line.find("]")
            url_start = line.find("(") + 1
            url_end = line.find(")")

            title = line[title_start:title_end]
            url = line[url_start:url_end]

            # Extract date from the line (assume the date is always at the start of the line)
            date = line.split(":")[0].strip() if ":" in line else "Unknown Date"
            date = date.replace("*","")
            # Remove asterisks around the date and calendar emoji
            date = date.replace("**", "")  # Remove any asterisks around the date

            
            # Format the output as Date → Title (hyperlinked) without the "Link" part
            formatted_summary += f"{date} → <{url}|{title}>\n"

        else:
            # If it's not a link, just add the line as is
            formatted_summary += line + "\n"

    return formatted_summary.strip().replace("**", "")

def send_message_to_slack(channel_id, message_text):
    """Sends a formatted message to Slack."""
    if not channel_id:
        print("❌ Cannot send message: Channel ID not found.")
        return

    try:
        response = client.chat_postMessage(channel=channel_id, text=message_text)
        print("✅ Message sent successfully to Slack!")
    except SlackApiError as e:
        print(f"❌ Error sending message: {e.response['error']}")

# Format the summary and send to Slack
formatted_summary = format_summary_for_slack(summary)
channel_id = get_channel_id(channel_name)

if formatted_summary and channel_id:
    send_message_to_slack(channel_id, formatted_summary)
