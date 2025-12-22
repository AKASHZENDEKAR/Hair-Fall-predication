import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import base64

@st.cache_data
def load_data():
    df = pd.read_csv('hairfall_problem3592.csv')
    return df

def preprocess_data(df):
    df = df.drop(columns=['Timestamp', 'What is your name ?'], errors='ignore')
    le = LabelEncoder()
    df['Do you have hair fall problem ?'] = le.fit_transform(df['Do you have hair fall problem ?'])
    df['What is your gender ?'] = le.fit_transform(df['What is your gender ?'])
    encode_cols = [
        'Is there anyone in your family having a hair fall problem or a baldness issue?',
        'Did you face any type of chronic illness in the past?',
        'Do you stay up late at night?',
        'Do you have any type of sleep disturbance?',
        'Do you think that in your area water is a reason behind hair fall problems?',
        'Do you use chemicals, hair gel, or color in your hair?',
        'Do you have anemia?',
        'Do you have too much stress',
        'What is your food habit'
    ]
    for col in encode_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df

def prepare_user_input(user_data, training_columns):
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df.reindex(columns=training_columns, fill_value=0)

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    st.set_page_config("Hair Fall & Bald Detection Dashboard", layout="wide")
    set_background("background.png")

    st.markdown("""
        <div style='background-color: #e8f6f3; padding: 30px 10px; text-align: center; border-radius:10px;'>
            <h1 style='color: #1e2f40;'>💇‍♀️ Discover the Root Cause of Hair Fall</h1>
            <h3 style='color: #117864;'>AI + Science-based Hair Assessment — Start Now!</h3>
        </div>
    """, unsafe_allow_html=True)

    df = preprocess_data(load_data())
    X = df.drop('Do you have hair fall problem ?', axis=1)
    y = df['Do you have hair fall problem ?']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, _ = train_logistic_regression(X_train, X_test, y_train, y_test)

    st.markdown("<h2 style='color:#2c3e50;'>📝 1. Take the Survey</h2>", unsafe_allow_html=True)
    with st.form("survey"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            age = st.slider('Age', 15, 100, 24)
            family_history = st.radio('Family history of hair fall?', ['Yes', 'No'])
            chronic_illness = st.radio('Chronic illness in past?', ['Yes', 'No'])
            stay_up_late = st.radio('Stay up late at night?', ['Yes', 'No'])
        with col2:
            sleep_disturbance = st.radio('Sleep disturbance?', ['Yes', 'No'])
            water_quality = st.radio('Is water a reason?', ['Yes', 'No'])
            use_chemicals = st.radio('Use hair chemicals?', ['Yes', 'No'])
            anemia = st.radio('Do you have anemia?', ['Yes', 'No'])
            stress = st.radio('Too much stress?', ['Yes', 'No'])
            food_habit = st.selectbox('Food habit', ['Nutritious', 'Junk', 'Both'])
        st.markdown("<h2 style='color:#2c3e50;'>📷 2. Upload or Capture a Scalp Image</h2>", unsafe_allow_html=True)
        input_method = st.radio("Choose input method:", ["Upload", "Camera"])
        image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png']) if input_method == "Upload" else st.camera_input("Take a photo")
        submitted = st.form_submit_button("🔍 Predict Hair Fall & Baldness")

    if submitted:
        if image is None:
            st.warning("Please upload or capture an image to continue.")
            return

        user_input = {
            'What is your gender ?': 1 if gender == 'Male' else 0,
            'What is your age ?': age,
            'Is there anyone in your family having a hair fall problem or a baldness issue?': 1 if family_history == 'Yes' else 0,
            'Did you face any type of chronic illness in the past?': 1 if chronic_illness == 'Yes' else 0,
            'Do you stay up late at night?': 1 if stay_up_late == 'Yes' else 0,
            'Do you have any type of sleep disturbance?': 1 if sleep_disturbance == 'Yes' else 0,
            'Do you think that in your area water is a reason behind hair fall problems?': 1 if water_quality == 'Yes' else 0,
            'Do you use chemicals, hair gel, or color in your hair?': 1 if use_chemicals == 'Yes' else 0,
            'Do you have anemia?': 1 if anemia == 'Yes' else 0,
            'Do you have too much stress': 1 if stress == 'Yes' else 0,
            'What is your food habit': 0 if food_habit == 'Nutritious' else 1 if food_habit == 'Junk' else 2
        }

        user_df = prepare_user_input(user_input, X.columns)
        survey_pred = model.predict(user_df)[0]

        try:
            cnn_model = load_model("cnn_trained_model.h5", compile=False)
            processed_img = preprocess_image(image)
            pred = cnn_model.predict(processed_img)[0][0]
            bald_pred = 1 if pred <= 0.5 else 0
        except Exception as e:
            st.error(f"CNN Prediction Failed: {e}")
            return

        if bald_pred == 1:
            final_status = 'Advanced Hair Loss'
        elif bald_pred == 0 and survey_pred == 1:
            final_status = 'Partial Stage of Hair Loss'
        else:
            final_status = 'No Baldness'

        color = {
            'Advanced Hair Loss': 'red',
            'No Baldness': 'green',
            'Partial Stage of Hair Loss': 'orange'
        }[final_status]

        st.markdown(f"""
        <h1 style='text-align:center; color:{color}; font-size: 45px;'>
        ✅ Prediction Result: {final_status}
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2 style='color:#8e44ad;'>💡 Personalized Recommendations</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        if final_status == 'Advanced Hair Loss':
            image_base64 = get_base64_image("image2.png")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <ul style='font-size:18px;'>
                    <li>🧘 Practice stress reduction (Yoga/Meditation)</li>
                    <li>💤 Maintain 7–8 hours of sleep</li>
                    <li>💊 Iron-rich food: spinach, lentils, etc.</li>
                    <li>🌿 Use herbal and chemical-free hair products</li>
                </ul>
                <h3 style='color:#c0392b;'>🛄 Recommended Hair Oils</h3>
                <ul>
                    <li><strong>Ashtaksha AyurGrow Hair Oil - ₹300 <del>₹330</del></strong></li>
                    <li><strong>Ashtaksha AyurThera Hair Oil - ₹450 <del>₹500</del></strong></li>
                </ul>
                <h3 style='color:#2980b9;'>🛄 Recommended Shampoos</h3>
                <ul>
                    <li><strong>Ashtaksha AyurGrow Shampoo - ₹340 <del>₹400</del></strong></li>
                </ul>
                <h4 style='color:green;'>🛒 Combo Offer Value: ₹1090 <del>₹1230</del> — You Save ₹140!</h4>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='border: 2px solid #8e44ad; padding: 8px; border-radius: 12px; background-color: #fefefe; text-align:center;'>
                    <img src='data:image/png;base64,{image_base64}' style='max-width:100%; height:auto; max-height:450px; border-radius:10px;'/>
                    <p style='font-weight: bold; margin-top: 10px;'>🛄 Hair Care Kit</p>
                </div>
                """, unsafe_allow_html=True)

        elif final_status == 'Partial Stage of Hair Loss':
            image_base64 = get_base64_image("image1.png")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <ul style='font-size:18px;'>
                    <li>💤 Get enough quality sleep</li>
                    <li>🧘 Reduce stress with daily relaxation</li>
                    <li>🥗 Improve nutrition (more greens & protein)</li>
                    <li>🚿 Avoid daily hot water hair washes</li>
                </ul>
                <h4 style='color:#e67e22;'>⚠️ Early signs of hair issues — take care now!</h4>
                <h3>🛄 Recommended Hair Oil</h3>
                <ul>
                    <li><strong>Ashtaksha AyurGrow Hair Oil - ₹340 <del>₹400</del></strong></li>
                </ul>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='border: 2px solid #f39c12; padding: 8px; border-radius: 12px; background-color: #fefefe; text-align:center;'>
                    <img src='data:image/png;base64,{image_base64}' style='max-width:100%; height:auto; max-height:450px; border-radius:10px;'/>
                    <p style='font-weight: bold; margin-top: 10px;'>🛄 Hair Care Kit</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.success("🎉 You're in good condition! Maintain a healthy lifestyle.")
            st.markdown("""
            <ul style='font-size:18px;'>
                <li>🥦 Stick to a nutritious balanced diet</li>
                <li>💆 Oil your hair once a week</li>
                <li>🛄 Use mild herbal hair products</li>
                <li>🧘 Stay calm & manage stress</li>
            </ul>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()



