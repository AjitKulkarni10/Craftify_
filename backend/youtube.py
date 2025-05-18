import requests
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv

load_dotenv()

# topic = "Cloud Computing Basics"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_video_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"

def search_youtube(search_query):
    search_query = search_query + " in English"
    encoded_query = requests.utils.quote(search_query)

    url = (
        f"https://www.googleapis.com/youtube/v3/search"
        f"?key={YOUTUBE_API_KEY}"
        f"&q={encoded_query}"
        f"&videoDuration=medium"
        f"&videoEmbeddable=true"
        f"&type=video"
        f"&maxResults=3"
        # f"&order=viewCount"
        # f"&publishedAfter=2023-01-01T00:00:00Z"

    )

    response = requests.get(url)

    if response.status_code != 200:
        print("Status code:", response.status_code)
        print("API request failed:", response.text)
        return None

    data = response.json()

    if "items" not in data or len(data["items"]) == 0:
        print("No video items found.")
        return None

    return data["items"][0]["id"]["videoId"]

def get_transcript(video_id):
    transcript_arr = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join([entry["text"] for entry in transcript_arr])
    return transcript.replace("\n", "")

# Get video ID and URL
# video_url = get_video_url(search_youtube(topic))