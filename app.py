import streamlit as st
import re
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import emoji
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer

import helper
import preprocessor
import seaborn as sns
st.set_page_config(layout="wide")


st.sidebar.title('Whatsapp Chat Analyzer')
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    #st.text(data)
    df=preprocessor.preprocessor(data)
    

    #Fetching of users
    user_list=df['User'].unique().tolist()
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user=st.sidebar.selectbox("Select User",user_list)
    system_messages = [
        "Media omitted",
        "image omitted",
        "This message was deleted",
        "added", "removed", "left",
        "changed the subject",
        "changed this group’s icon",
        "video omitted",
        "Messages and calls are end-to-end encrypted"
    ]

    # Remove system messages from df['Message ']
    df = df[~df["Message"].str.contains("|".join(system_messages), na=False, case=False)]
    unwanted_phrases = [
        "joined using this group's invite link",
        "left",
        "changed the group name",
        "changed this group’s icon",
        "added",
        "removed",
        "You're now an admin",
        "Media omitted",
        "This message was deleted",
        "deleted this message"
    ]

    # Remove rows where 'Message' contains any of the unwanted phrases
    df = df[~df["Message"].str.contains('|'.join(unwanted_phrases), na=False)]
    def clean_usernames(User):
        if re.search(r"changed the subject|added|removed|created this group", User):
            return None
        return User

    df['User'] = df['User'].apply(clean_usernames)
    df.dropna(subset=['User'], inplace=True)
    st.dataframe(df)

    if st.sidebar.button("show Analysis of {}".format(selected_user)):
        num_messages, num_words, messages, words, num_links, links = helper.fetch_length(selected_user, df)
        col1, col2 ,col3= st.columns(3)
        with col1:
            st.header("Total Messages={}".format(num_messages))
        with col2:
            st.header("Total Words={}".format(num_words))
        with col3:
            st.header("Number of links shared={}".format(num_links))
    if st.sidebar.button("Show Content of {}".format(selected_user)):
        num_messages, num_words, messages, words, num_links, links = helper.fetch_length(selected_user, df)
        co1, co2 = st.columns(2)
        with co1:
            st.header("Messages of {}".format(selected_user))
            st.dataframe(messages)
        with co2:
            st.header("Links shared by {}".format(selected_user))
            st.dataframe(links)

    if st.sidebar.button("Rate of messages of {}".format(selected_user)):
        new_df,df1=helper.timeline_users(selected_user,df)
        fig, ax = plt.subplots()
        ax.plot(new_df['time'], new_df['Message'],color='green')  # Added markers
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Messages")
        ax.set_title("Messages Over Time")
        plt.xticks(rotation=90)  # Rotated x-axis labels for better visibili
        st.pyplot(fig)
        fig1, ax1= plt.subplots()
        ax1.plot(df1['Date'],df1['Message'],color='black' )  # Added markers
        ax1.set_xlabel("Dates")
        ax1.set_ylabel("Number of Messages")
        ax1.set_title("Messages Over Time")
        plt.xticks(rotation=90)
        plt.figure(figsize=(8,8))# Rotated x-axis labels for better visibili
        st.pyplot(fig1)

    if st.sidebar.button("Most used words by {}".format(selected_user)):
        st.header('Words used by {}'.format(selected_user))
        #new_df = helper.most_common_words(selected_user,df)
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        new_df,filtered_words = helper.most_common_words(selected_user, df)
        st.dataframe(new_df)
    if st.sidebar.button("Most used emojis by {}".format(selected_user)):
        st.header('Emojis used by {}'.format(selected_user))
        col1, col2 = st.columns(2)
        with col1:
            df_emojis,top_emojis=helper.most_common_emojis(selected_user, df)
            st.dataframe(df_emojis)
        with col2:
            f_emoji = pd.Series(["😂", "😂", "❤️", "👍", "😂", "🙏", "😊", "😊", "😊", "🔥", "🔥", "🔥", "🔥", "😍", "🎉", "💯"])
            # Convert MultiIndex to a flat list
            # Convert index and values to lists (Fix MultiIndex issue)
            emoji_counts = df_emojis.value_counts()

            # Streamlit App Layout
            st.title("📊 Emoji Usage Distribution")
            st.write("### Pie Chart of Emoji Frequency")

            # Plot Pie Chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(emoji_counts, labels=emoji_counts.index, autopct='%1.1f%%', startangle=140,
                   textprops={'fontsize': 14})
            ax.set_title("Emoji Usage Distribution", fontsize=16)

            # Display plot in Streamlit
            st.pyplot(fig)
    if st.sidebar.button("Most busy Days & Months of  {}".format(selected_user)):
        st.title('Most busy Days')
        df1,df2=helper.Most_busy_timeline(selected_user, df)
        fig,ax=plt.subplots()
        ax.bar(df1['Day_name'], df1['Message'],color='green')
        st.pyplot(fig)
        st.title('Most busy Months')
        fig,ax=plt.subplots()
        ax.bar(df2['Month'], df2['Message'],color='yellow')
        st.pyplot(fig)

    if st.sidebar.button("Show Sentiment Analysis of {}".format(selected_user)):
        df1,positive,negative,neutral=helper.sentiment_analysis(selected_user, df)
        # Display Results
        st.subheader("📌 Sentiment Analysis Results")
        st.write(f"**Positive Messages:** {positive}")
        st.write(f"**Neutral Messages:** {neutral}")
        st.write(f"**Negative Messages:** {negative}")

        # Display Chat Data with Sentiment
        st.subheader("📝 Processed Chat Messages")
        st.dataframe(df1)

        # Visualization
        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        labels = ["Positive", "Neutral", "Negative"]
        values = [positive, neutral, negative]
        ax.bar(labels, values, color=['green', 'gray', 'red'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Message Count")
        ax.set_title("WhatsApp Chat Sentiment Analysis")
        st.pyplot(fig)
        sia = SentimentIntensityAnalyzer()

        user_sentiments = {}

        # Process Chat Messages
        for user, message in zip(df['User'], df['Message']):
            score = sia.polarity_scores(str(message))  # Ensure message is string
            sentiment = score['compound']  # Compound Score (-1 to +1)

            if user not in user_sentiments:
                user_sentiments[user] = []
            user_sentiments[user].append(sentiment)

        # Convert to DataFrame
        user_df = pd.DataFrame({
            "User": list(user_sentiments.keys()),
            "Avg Sentiment": [sum(scores) / len(scores) for scores in user_sentiments.values()]
        })

        # Normalize Data
        scaler = StandardScaler()
        user_df["Sentiment Scaled"] = scaler.fit_transform(user_df[["Avg Sentiment"]])

        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        user_df["Cluster"] = kmeans.fit_predict(user_df[["Sentiment Scaled"]])

        # Display Data
        st.write("### 🏷️ Sentiment Analysis Results:")
        st.dataframe(user_df)

        # Visualization
        st.write("### 📊 Sentiment Clustering:")
        fig, ax = plt.subplots(figsize=(20,15))
        sns.barplot(x=user_df["User"], y=user_df["Avg Sentiment"], hue=user_df["Cluster"], palette="coolwarm", ax=ax)
        plt.xlabel("User")
        plt.ylabel("Average Sentiment Score")
        plt.title("User Sentiment Clustering")
        plt.xticks(rotation=90)
        plt.legend(title="Cluster")
        st.pyplot(fig)
    if st.sidebar.button("Show most available users"):
            num_messages,num_words,messages,words,num_links,links=helper.fetch_length(selected_user,df)
            col1, col2 = st.columns(2)
            with col1:
                x,new_df=helper.fetch_available_users(df)
                fig, ax = plt.subplots()
                ax.bar(x.index,x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.dataframe(new_df)

