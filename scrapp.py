import streamlit as st
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import altair as alt
import base64

def main():
    st.title("Welcome to the Review Scraper and AI Model")
    # Your main app code here

# Encode your local image to Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Replace with your local image path
encoded_image = image_to_base64("C:\\Users\\ahmed\\Downloads\\0130936b2eea1b178b041251c974c013.jpg")

# Set up Streamlit page config
st.set_page_config(page_title="Review Scraper & Emotion Detector", layout="wide")

# Add the background image with CSS
st.markdown(f"""
    <style>
    body {{
        background-image: url('data:image/jpg;base64,{encoded_image}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .reportview-container {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        animation: fadeIn 2s ease-in-out;
    }}
    .sidebar .sidebar-content {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        animation: fadeIn 2s ease-in-out;
    }}
    .stButton>button {{
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #0056b3;
        transform: scale(1.05);
    }}
    .stTextInput>div>input {{
        border: 2px solid #007bff;
        border-radius: 5px;
    }}
    .stSpinner {{
        border: 4px solid #007bff;
    }}
    .stDataFrame {{
        animation: fadeIn 2s ease-in-out;
    }}
    </style>
""", unsafe_allow_html=True)

# Load your AI model and CountVectorizer
@st.cache_resource
def load_model():
    with open('xgb_emotion_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Create the WebDriver
def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    service = Service('C:\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe')  # Update with your actual path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Scrape reviews from a URL
def scrape_reviews(url, max_reviews=1000):
    driver = create_webdriver()
    driver.get(url)
    reviews = []
    last_review_count = 0
    scroll_attempts = 0

    css_selectors = [
        ".jdgm-rev__body p",
        ".text.show-more__control",
        ".review-text",
        ".review-content",
        "div.comment-body",
        ".review-body",
        ".ugc-review",
        ".comment",
        "article",
        "blockquote",
    ]

    try:
        while len(reviews) < max_reviews:
            found = False
            for selector in css_selectors:
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        for element in elements:
                            review_text = element.text.strip()
                            if review_text and review_text not in reviews:
                                reviews.append(review_text)
                                if len(reviews) >= max_reviews:
                                    break
                        found = True
                        break
                except Exception as e:
                    continue

            if not found:
                break

            if len(reviews) == last_review_count:
                scroll_attempts += 1
            else:
                scroll_attempts = 0

            last_review_count = len(reviews)

            if scroll_attempts >= 3:
                break

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        driver.quit()
    return reviews

# Predict emotions for the reviews
def predict_emotions(reviews):
    review_vectors = vectorizer.transform(reviews)
    predictions = model.predict(review_vectors)
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return [emotion_labels[pred] for pred in predictions]

# App UI
st.title("📝 Universal Review Scraper with Emotion Detection")

url = st.text_input("🔗 Enter the URL of the reviews page:")
if st.button("Scrape Reviews"):
    if url:
        with st.spinner("Scraping reviews..."):
            try:
                reviews = scrape_reviews(url)
                if reviews:
                    st.write(f"Scraped {len(reviews)} reviews.")

                    emotions = predict_emotions(reviews)

                    # Combine reviews and emotions into a DataFrame
                    data = {"Review": reviews, "Emotion": emotions}
                    df = pd.DataFrame(data)

                    # Define colors for each emotion
                    emotion_colors = {
                        "sadness": "#FF6F61",
                        "joy": "#FFD700",
                        "love": "#FF69B4",
                        "anger": "#DC143C",
                        "fear": "#8A2BE2",
                        "surprise": "#40E0D0"
                    }

                    # Apply conditional formatting to color reviews based on emotions
                    def highlight_emotions(row):
                        return ['background-color: {}'.format(emotion_colors[row.Emotion])] * len(row)

                    styled_df = df.style.apply(highlight_emotions, axis=1)

                    # Display the DataFrame as a table with colored rows
                    st.write("### Reviews and Emotions")
                    st.dataframe(styled_df)

                    # Display the emotion distribution as a custom bar chart
                    emotion_counts = pd.Series(emotions).value_counts().reset_index()
                    emotion_counts.columns = ['Emotion', 'Count']

                    # Create a more advanced Altair chart
                    chart = alt.Chart(emotion_counts).mark_bar(
                        cornerRadiusTopLeft=10,
                        cornerRadiusTopRight=10
                    ).encode(
                        x=alt.X('Emotion:O', title='Emotion'),
                        y=alt.Y('Count:Q', title='Count'),
                        color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category20b')),
                        tooltip=['Emotion', 'Count']
                    ).properties(
                        title='Emotion Distribution',
                        width=600,
                        height=400
                    ).configure_title(
                        fontSize=24,
                        font='Helvetica',
                        anchor='start',
                        color='#007bff'
                    ).configure_axis(
                        labelFontSize=14,
                        titleFontSize=16,
                        grid=False
                    ).configure_view(
                        strokeWidth=0
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No reviews found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a URL.")
