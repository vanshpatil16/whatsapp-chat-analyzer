from urlextract import URLExtract
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import pandas as pd
import emoji
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
def extract_urls(text):
    extractor = URLExtract()
    return extractor.find_urls(text)


def fetch_length(selected_user,df):
    if selected_user=="Overall":
        word = []
        link = []
        for message in df['Message']:
            word.extend(message.split())
            link.extend(extract_urls(message))
        return df.shape[0],len(word),df['Message'],word,len(link),link
    else:

        new_df=df[df["User"]==selected_user]
        word=[]
        link=[]
        for message in new_df['Message']:
            word.extend(message.split())
            link.extend(extract_urls(message))
        return new_df.shape[0],len(word),new_df['Message'],word,len(link),pd.DataFrame(link)
def fetch_available_users(df):
    x=df['User'].value_counts().head(10)
    new_df=round((df['User'].value_counts() / df.shape[0]) * 100).reset_index().rename(columns={'count': 'percentage'})
    return x,new_df
def create_wordcloud(selected_user,df):
    if selected_user!='Overall':
        df=df[df["User"]==selected_user]
    new_df,filtered_words=most_common_words(selected_user,df)
    # Convert the list to a space-separated string
    text = " ".join(filtered_words)

    # Generate the word cloud
    wc = WordCloud(max_font_size=80, width=800, height=600, background_color='white')
    df_wc = wc.generate(text)
    return df_wc
def most_common_words(selected_user,df):

    if selected_user !='Overall':
        new_df2 = df[ selected_user==df["User"]]
    else:
        new_df2= df
    keywords = ["joined using this group", "left the group", "was added", "added", "image omitted"]

    # Filter out rows containing unwanted messages
    df_cleaned = new_df2[~new_df2["Message"].str.contains("|".join(keywords), na=False, case=False)]

    # Display cleaned DataFrame
    filtered_words = []
    for message in df_cleaned['Message']:
        filtered_words.extend(message.split())

    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # Define stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(["image", "omitted"])

    # Remove stop words
    new_filtered_words = [word for word in filtered_words if word not in stop_words]
    new_df=pd.DataFrame(new_filtered_words)
    new_df1=new_df.value_counts().head(20)
    return new_df1,new_filtered_words
def most_common_emojis(selected_user,df):
    if selected_user!='Overall':
        df = df[ selected_user==df["User"]]
        emojis = []
        for message in df['Message']:
            emojis.extend([c for c in message if emoji.is_emoji(c)])
        df_emojis = pd.DataFrame(emojis)
        df_emojis = pd.DataFrame(df_emojis.value_counts())
        return df_emojis,df_emojis.head(10)
def timeline_users(selected_user,df):
    if selected_user!='Overall':
        df = df[df["User"]==selected_user]

    df["Date & Time"] = pd.to_datetime(df["Date & Time"], format="%d/%m/%y %H:%M:%S")

    # Create a new "Date" column
    df["Date"] = df["Date & Time"].dt.date  # Extract only the date
    timeline = df.groupby(['Month', 'Year'])['Message'].count().reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['Month'][i] + "-" + str(timeline['Year'][i]))
    timeline['time'] = time
    df1 = df.groupby(['Date'])['Message'].count().reset_index()

    return timeline,df1
def Most_busy_timeline(selected_user,df):
    if selected_user!='Overall':
        df = df[df["User"]==selected_user]
    t,df=timeline_users(selected_user,df)
    df["Date"] = pd.to_datetime(df["Date"])

    #  Add a new column with day names
    df["Day_name"] = df["Date"].dt.day_name()
    df1=df['Day_name'].value_counts().reset_index().rename(columns={'count': 'Message'})
    df2 = df['Date'].dt.month_name().value_counts().reset_index().rename(columns={'count': 'Message','Date':'Month'})
    return df1,df2
def sentiment_analysis(selected_user,df):
    if selected_user!='Overall':
        df = df[df["User"]==selected_user]
    import re
    import nltk
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st
    from nltk.sentiment import SentimentIntensityAnalyzer

    # Download NLTK Vader Lexicon
    nltk.download('vader_lexicon')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Streamlit App UI
    st.title("📊 WhatsApp Chat Sentiment Analysis")
    st.write("Upload a WhatsApp chat export file (.txt) to analyze its sentiment.")

    positive, neutral, negative = 0, 0, 0
    processed_messages = []

        # Process Chat Messages
    for message in df['Message']:
        # Perform sentiment analysis
        score = sia.polarity_scores(message)
        sentiment = "Neutral"
        if score['compound'] >= 0.05:
            sentiment = "Positive"
            positive += 1
        elif score['compound'] <= -0.05:
            sentiment = "Negative"
            negative += 1
        else:
            neutral += 1

        processed_messages.append([message, sentiment])


        # Convert to DataFrame
    df1 = pd.DataFrame(processed_messages, columns=["Message", "Sentiment"])
    return df1,positive,negative,neutral














