import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Application Backend

# To load medicine-dataframe from pickle in the form of dictionary
medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
reasons_dict = pickle.load(open('uses.pkl', 'rb'))
reasons = pd.DataFrame(reasons_dict)

# To load similarity-vector-data from pickle in the form of dictionary
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load descriptions of medicines
medicine_descriptions = pd.Series(medicines_dict['Description'], index=medicines_dict['Drug_Name'])

# Convert to lowercase only for string values
def custom_preprocessor(text):
    return text.lower() if isinstance(text, str) else str(text)

# Vectorize the medicine descriptions
vectorizer = TfidfVectorizer(stop_words='english', preprocessor=custom_preprocessor)
medicine_vectors = vectorizer.fit_transform(medicine_descriptions)

def recommend(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = []
    for i in medicines_list:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
    return recommended_medicines

def reason(medicine):
    medicine_index = reasons[reasons['Drug_Name'] == medicine]
    medicine_use = medicine_index['Description'].values[0]
    return medicine_use

def search_best_medicine(illness_description):
    # Vectorize the input illness description
    illness_vector = vectorizer.transform([illness_description])

    # Calculate cosine similarity between illness description and medicine descriptions
    similarity_scores = cosine_similarity(illness_vector, medicine_vectors).flatten()

    # Rank medicines based on similarity
    ranked_medicines = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])

    # Return the top recommended medicines
    recommended_medicines = []
    for i in ranked_medicines[:5]:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)

    return recommended_medicines

# Application Frontend

# Set page configuration
st.set_page_config(
    
    page_title="KnowMeds - Your Medical Companion",
    page_icon=":pill:",
    layout="wide"
)


# Colored background with border for the title
st.markdown(
    """
    <style>
        /* Overall styling */
        body, .stApp {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title section */
        .title-text {
            background-color: #1abc9c;
            color: white;
            padding: 1.2rem;
            border-radius: 1rem;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }

        /* Section headers */
        .section-heading, .illness-heading, .doctor-heading {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-top: 2rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }

        .section-heading {
            background-color: #3498db;
            border: 2px solid #2c3e50;
        }

        .illness-heading {
            background-color: #e74c3c;
            border: 2px solid #c0392b;
        }

        .doctor-heading {
            background-color: #9b59b6;
            border: 2px solid #8e44ad;
        }

        /* Form fields */
        textarea, .stTextInput > div > input {
            border-radius: 0.5rem !important;
            border: 1px solid #bdc3c7;
            padding: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            margin-top: 1rem;
            border-radius: 0.6rem;
            background-color: #2980b9;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #1f618d;
        }

        /* Expander content */
        .stExpander {
            background-color: #ecf0f1 !important;
            border-radius: 0.5rem;
            padding: 1rem;
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background-color: #bdc3c7;
            margin: 2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-text'>KnowMeds</h1>", unsafe_allow_html=True)

# User instructions
st.info("Select a medicine and click 'Recommend Medicine' to get alternatives. Describe an illness and click 'Find Best Medicine' to get recommendations based on the illness.")

# Colored background with border for section headings
st.markdown(
    """
    <style>
        /* Overall styling */
        body, .stApp {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title section */
        .title-text {
            background-color: #1abc9c;
            color: white;
            padding: 1.2rem;
            border-radius: 1rem;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }

        /* Section headers */
        .section-heading, .illness-heading, .doctor-heading {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-top: 2rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }

        .section-heading {
            background-color: #3498db;
            border: 2px solid #2c3e50;
        }

        .illness-heading {
            background-color: #e74c3c;
            border: 2px solid #c0392b;
        }

        .doctor-heading {
            background-color: #9b59b6;
            border: 2px solid #8e44ad;
        }

        /* Form fields */
        textarea, .stTextInput > div > input {
            border-radius: 0.5rem !important;
            border: 1px solid #bdc3c7;
            padding: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            margin-top: 1rem;
            border-radius: 0.6rem;
            background-color: #2980b9;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #1f618d;
        }

        /* Expander content */
        .stExpander {
            background-color: #ecf0f1 !important;
            border-radius: 0.5rem;
            padding: 1rem;
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background-color: #bdc3c7;
            margin: 2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Searchbox for selecting medicine
st.markdown("<h2 class='section-heading'>Select Medicine</h2>", unsafe_allow_html=True)
selected_medicine_name = st.selectbox(
    'Type your medicine name whose alternative is to be recommended',
    medicines['Drug_Name'].values
)

# Recommendation Program
if st.button('Recommend Medicine'):
    recommendations = recommend(selected_medicine_name)
    if recommendations:
        st.subheader('Top Recommended Medicines')
        for i, medicine in enumerate(recommendations, start=1):
            st.write(f"{i}. {medicine}")
            st.write(f"Purchase at [PharmEasy](https://pharmeasy.in/search/all?name={medicine})")
    else:
        st.warning("No alternative medicines found.")

# Uses of Selected Medicine
if st.button('Uses of Selected Medicine'):
    uses = reason(selected_medicine_name)
    uses = ' '.join(uses)
    st.subheader('Uses of Selected Medicine')
    st.write(uses)

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Colored background with border for illness description
st.markdown(
    """
    <style>
        /* Overall styling */
        body, .stApp {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title section */
        .title-text {
            background-color: #1abc9c;
            color: white;
            padding: 1.2rem;
            border-radius: 1rem;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }

        /* Section headers */
        .section-heading, .illness-heading, .doctor-heading {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-top: 2rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }

        .section-heading {
            background-color: #3498db;
            border: 2px solid #2c3e50;
        }

        .illness-heading {
            background-color: #e74c3c;
            border: 2px solid #c0392b;
        }

        .doctor-heading {
            background-color: #9b59b6;
            border: 2px solid #8e44ad;
        }

        /* Form fields */
        textarea, .stTextInput > div > input {
            border-radius: 0.5rem !important;
            border: 1px solid #bdc3c7;
            padding: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            margin-top: 1rem;
            border-radius: 0.6rem;
            background-color: #2980b9;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #1f618d;
        }

        /* Expander content */
        .stExpander {
            background-color: black !important;
            border-radius: 0.5rem;
            padding: 1rem;
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background-color: #bdc3c7;
            margin: 2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 class='illness-heading'>Describe Illness</h2>", unsafe_allow_html=True)

# Searchbox for entering illness description
illness_description = st.text_area("Describe the illness:", height=150)
illness_description = illness_description.lower()


# Recommendation Program for best medicine based on illness description
if st.button('Find Best Medicine'):
    recommendations = search_best_medicine(illness_description)
    if recommendations:
        st.subheader('Top Recommended Medicines and Their Uses')
        for i, medicine in enumerate(recommendations, start=1):
            # Create an expander for each medicine
            with st.expander(f"{i}. {medicine} - Click to view description and uses"):
                # Show the uses of the selected medicine
                uses = reason(medicine)
                uses = ' '.join(uses)
                st.write(uses)

         
# Image load
image = Image.open('logo.jpeg')
st.image(image, caption='Your Medical Companion', use_container_width=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #3498db;
        color: white;
        text-align: center;
        padding: 0.8rem 0;
        font-size: 14px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .footer a {
        color: white;
        margin: 0 10px;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        © 2025 KnowMeds Team | 
        <a href="https://twitter.com/knowmeds" target="_blank">Twitter</a> | 
        <a href="https://www.facebook.com/knowmeds" target="_blank">Facebook</a> | 
        <a href="https://www.linkedin.com/company/knowmeds" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
