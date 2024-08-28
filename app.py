import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import mysql.connector
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load the fine-tuned model and tokenizer
model_path = 'D:/DATA_VentureX/LLM_Content_generation_data/'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# MySQL connection setup
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='admin',
    database='youtube_data'
)

# Fetch data from MySQL
query = """
    SELECT DISTINCT video_id, transcription 
    FROM (
        SELECT video_id, transcription 
        FROM youtube_audio_transcriptions 
        WHERE video_id IS NOT NULL AND transcription IS NOT NULL

        UNION ALL

        SELECT video_id, transcription 
        FROM youtube_transcriptions 
        WHERE video_id IS NOT NULL AND transcription IS NOT NULL
    ) AS combined
    GROUP BY video_id, transcription;
"""
df = pd.read_sql(query, conn)

# Fetch video details
def get_video_details(video_ids):
    video_data = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            request = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(batch_ids)
            )
            response = request.execute()
            for item in response['items']:
                video_data.append({
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0)),
                    'published_at': item['snippet']['publishedAt']
                })
        except HttpError as e:
            print(f"An error occurred: {e}")
            time.sleep(5)
            continue
    return pd.DataFrame(video_data)

video_ids = df['video_id'].tolist()
video_details_df = get_video_details(video_ids)
df = pd.merge(df, video_details_df, on='video_id')

# Streamlit UI
st.title("YouTube Content Generation and Analytics")

# Content Generation
st.header("Content Generation")
prompt = st.text_area("Enter a prompt:")
if st.button("Generate Content"):
    generated = generator(prompt, max_length=200, num_return_sequences=1)
    st.write(generated[0]['generated_text'])

# Visualization
st.header("YouTube Video Analytics")
st.write("Visualizing engagement metrics...")
fig, ax = plt.subplots()
sns.barplot(x="title", y="views", data=df, ax=ax)
ax.set_title("Views by Video")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

# Example: Show the video with the maximum engagement
def get_max_engagement_video(df):
    return df[df['likes'] == df['likes'].max()]

if st.button("Show Top Engagement Video"):
    top_video = get_max_engagement_video(df)
    st.write(top_video)
