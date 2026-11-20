
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import sqlite3
import pandas as pd
from datetime import datetime
from PIL import Image
from rag_engine import generate_answer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO

DB_PATH =r"C:\Users\SIDDHI\tomatocare.db"

def get_plants():
    conn = sqlite3.connect(DB_PATH)

    plants_df = pd.read_sql_query(
        "SELECT * FROM plants ORDER BY plant_id DESC",
        conn
    )

    conn.close()
    return plants_df


def get_scan_history(plant_id):
    conn = sqlite3.connect(DB_PATH)

    history_df = pd.read_sql_query(
        """
        SELECT scan_id, plant_id, disease, confidence, scan_date
        FROM scans
        WHERE plant_id = ?
        ORDER BY scan_id DESC
        """,
        conn,
        params=(plant_id,)
    )

    conn.close()
    return history_df

def compare_plant_health(plant_id):
    conn = sqlite3.connect(DB_PATH)

    history_df = pd.read_sql_query(
        """
        SELECT disease, confidence, scan_date
        FROM scans
        WHERE plant_id = ?
        ORDER BY scan_id DESC
        LIMIT 2
        """,
        conn,
        params=(plant_id,)
    )

    conn.close()
    return history_df

def save_scan(plant_id, disease, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans
        (plant_id, disease, confidence, scan_date)
        VALUES (?, ?, ?, ?)
    """, (
        plant_id,
        disease,
        float(confidence),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def get_individual_plant_status():
    conn = sqlite3.connect(DB_PATH)

    status_df = pd.read_sql_query("""
        SELECT
            p.plant_id,
            p.plant_name,
            p.plant_location,
            p.plant_type,
            s.disease AS latest_disease,
            s.confidence AS confidence,
            s.scan_date AS last_scan
        FROM plants p
        LEFT JOIN scans s
            ON s.scan_id = (
                SELECT MAX(s2.scan_id)
                FROM scans s2
                WHERE s2.plant_id = p.plant_id
            )
        ORDER BY p.plant_id DESC
    """, conn)

    conn.close()

    status_df["health_status"] = status_df["latest_disease"].apply(
        lambda disease:
            "Not Scanned" if pd.isna(disease)
            else "Healthy" if disease == "healthy"
            else "Disease Detected"
    )

    return status_df

def generate_pdf_report(
    plant_name,
    plant_id,
    disease,
    confidence,
    scan_date,
    remedies
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph("TomatoCare AI - Disease Report", styles["Title"])
    )

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        f"<b>Plant Name:</b> {plant_name}", styles["Normal"]
    ))

    content.append(Paragraph(
        f"<b>Plant ID:</b> {plant_id}", styles["Normal"]
    ))

    content.append(Paragraph(
        f"<b>Detected Disease:</b> {disease.replace('_', ' ').title()}",
        styles["Normal"]
    ))

    content.append(Paragraph(
        f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]
    ))

    content.append(Paragraph(
        f"<b>Scan Date:</b> {scan_date}", styles["Normal"]
    ))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "Recommended Remedies", styles["Heading2"]
    ))

    for remedy in remedies:
        content.append(Paragraph(f"- {remedy}", styles["Normal"]))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "Note: This report contains an AI-based prediction, "
        "not a confirmed agricultural diagnosis.",
        styles["Italic"]
    ))

    doc.build(content)
    buffer.seek(0)

    return buffer



# Page settings
st.set_page_config(
    page_title="TomatoCare AI",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

/* Overall page background */
.stApp {
    background: linear-gradient(135deg, #E5EFE5 0%, #F2F5EA 55%, #E4EEE5 100%);
    color: #20382B;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123D2B 0%, #1C553B 100%);
    border-right: 1px solid #2C6849;
}

[data-testid="stSidebar"] * {
    color: #F3F8F1 !important;
}

/* Headings */
h1, h2, h3 {
    color: #174A32 !important;
    font-weight: 750 !important;
}

h1 {
    letter-spacing: -0.8px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #EDF5EC 100%);
    border: 1px solid #C9DDCA;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 5px 16px rgba(24, 70, 43, 0.09);
}

[data-testid="stMetricLabel"] {
    color: #52735C !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #174A32 !important;
    font-weight: 750;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #267A4B, #185C38);
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-weight: 650;
    box-shadow: 0 3px 9px rgba(24, 92, 56, 0.18);
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #104A2D;
    border: none;
    transform: translateY(-1px);
}

/* Input fields */
.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF;
    border: 1px solid #C6D9C8;
    border-radius: 9px;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF;
    border-radius: 9px;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #C9DDCA;
    border-radius: 13px;
    padding: 5px;
}

/* Charts and content containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 11px;
    border: 1px solid #D0E2D1;
}

/* Dividers */
hr {
    border-color: #C5D9C7;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.65);
    border: 1px dashed #8CAF91;
    border-radius: 13px;
    padding: 12px;
}

/* Sidebar radio navigation */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)






st.title("🍅 TomatoCare AI")
st.caption("AI-Powered Tomato Plant Health Monitoring System")

st.sidebar.title("🍅 TomatoCare AI")
st.sidebar.caption("Plant Health Assistant")


page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Individual Plant Health Status",
        "Disease Detection",
        "Plant Health History",
        "AI Assistant"
    ]
)

# INDIVIDUAL PLANT HEALTH STATUS
if page == "Individual Plant Health Status":

    st.title("Individual Plant Health Status")
    st.caption("Monitor the latest recorded condition of each registered tomato plant.")

    status_df = get_individual_plant_status()

    if status_df.empty:
        st.info("No plants registered yet. Please register a plant first.")

    else:
        # Summary metrics
        total = len(status_df)
        healthy = (status_df["health_status"] == "Healthy").sum()
        diseased = (status_df["health_status"] == "Disease Detected").sum()
        not_scanned = (status_df["health_status"] == "Not Scanned").sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Plants", total)
        col2.metric("Healthy", healthy)
        col3.metric("Disease Detected", diseased)
        col4.metric("Not Scanned", not_scanned)

        st.divider()
        st.subheader("Plant-wise Health Overview")

        for _, plant in status_df.iterrows():

            with st.container(border=True):

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(
                        f"### {plant['plant_name']} "
                        f"(Plant ID: {plant['plant_id']})"
                    )

                    st.write(
                        f"**Location:** {plant['plant_location'] or 'Not specified'}"
                    )
                    st.write(
                        f"**Plant Type:** {plant['plant_type'] or 'Tomato'}"
                    )

                    if pd.notna(plant["latest_disease"]):
                        st.write(
                            f"**Latest Detection:** {plant['latest_disease'].replace('_', ' ').title()}"
                        )
                        st.write(
                            f"**Model Confidence:** {plant['confidence']:.2%}"
                        )
                        st.write(
                            f"**Last Scan:** {plant['last_scan']}"
                        )
                    else:
                        st.write("**Latest Detection:** No scan recorded")

                with col2:
                    st.write("**Health Status**")

                    if plant["health_status"] == "Healthy":
                        st.success("Healthy")

                    elif plant["health_status"] == "Disease Detected":
                        st.error( "Disease Detected")

                    else:
                        st.warning(" Not Scanned")

    st.caption(
        "Note: Status is based on the latest AI scan, not a physical examination. "
        "Model confidence indicates prediction confidence, not plant recovery."
    )

            
    st.stop()

# PLANT HEALTH HISTORY
if page == "Plant Health History":

    st.title("Plant Health History")

    plants_df = get_plants()

    if plants_df.empty:
        st.info("Please register a plant first.")

    else:
        plant_options = {
            f"{row['plant_name']} (ID: {row['plant_id']})": row["plant_id"]
            for _, row in plants_df.iterrows()
        }

        selected_label = st.selectbox(
            "Select Plant",
            list(plant_options.keys())
        )

        selected_plant_id = plant_options[selected_label]

        history_df = get_scan_history(selected_plant_id)

        if history_df.empty:
            st.info("No scans recorded for this plant.")

        else:
            st.subheader("‹ Scan History")
            st.dataframe(history_df, use_container_width=True)

            history_df["scan_date"] = pd.to_datetime(
                history_df["scan_date"],
                errors="coerce"
            )

            history_df = history_df.dropna(subset=["scan_date"])

            if not history_df.empty:

                history_df["scan_day"] = (
                    history_df["scan_date"].dt.strftime("%Y-%m-%d")
                )

                trend_df = (
                    history_df.groupby(["scan_day", "disease"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )

                st.subheader("ˆ Disease Trend Over Time")
                st.write("Disease chart data:")
                st.write(trend_df)
                st.line_chart(trend_df)

            else:
                st.info("No valid scan dates available.")

    st.stop()


        
if page == "Dashboard":

    st.title(" Welcome to TomatoCare AI")
    st.write("AI-Powered Tomato Plant Health Monitoring")

    plants_df = get_plants()

    conn = sqlite3.connect(DB_PATH)

    scans_df = pd.read_sql_query(
        "SELECT * FROM scans",
        conn
    )

    conn.close()

    total_plants = len(plants_df)
    total_scans = len(scans_df)

    healthy_scans = len(
        scans_df[scans_df["disease"] == "healthy"]
    )

    disease_scans = total_scans - healthy_scans

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Registered Plants", total_plants)
    col2.metric(" Total Scans", total_scans)
    col3.metric("Healthy Scans", healthy_scans)
    col4.metric("âš ï¸ Disease Scans", disease_scans)

    st.subheader(" Individual Plant Monitoring")

    conn = sqlite3.connect(DB_PATH)

    plant_status_df = pd.read_sql_query("""
        SELECT
            p.plant_id,
            p.plant_name,
            s.disease AS latest_disease,
            s.confidence,
            s.scan_date
        FROM plants p
        LEFT JOIN scans s
        ON s.scan_id = (
            SELECT MAX(s2.scan_id)
            FROM scans s2
            WHERE s2.plant_id = p.plant_id
        )
        ORDER BY p.plant_id
    """, conn)

    conn.close()

    st.dataframe(
        plant_status_df,
        use_container_width=True
    )

    st.subheader("‹ Recent Scan Activity")
    st.subheader(" Disease Distribution")

    st.subheader("‹ Recent Scan Activity")

    if not scans_df.empty:
        st.dataframe(
            scans_df.sort_values(
                "scan_id", ascending=False
            ).head(5),
            use_container_width=True
        )
    else:
        st.info("No scan records available yet.")

    st.subheader(" Disease Distribution")

    if not scans_df.empty:
        disease_counts = scans_df["disease"].value_counts()
        st.bar_chart(disease_counts)
    else:
        st.info("No disease data available yet.")

# Model and dataset paths
MODEL_PATH = r"D:\Tomato Plant Disease\tomato_care_cnn.keras"
TRAIN_DIR = r"D:\Tomato Plant Disease\Dataset\train"

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.write("Loaded model:", MODEL_PATH)
st.write("Model output shape:", model.output_shape)

# Get class names in the same alphabetical order used during training
categories = [
    'bacterial_spot',
    'early_blight',
    'healthy',
    'late_blight',
    'leaf_mold',
    'mosaic_virus',
    'septoria_leaf_spot',
    'target_spot',
    'twospotted_spider_mite',
    'yellow_leaf_curl_virus'
]

remedies = {
    "bacterial_spot": [
        "Remove infected leaves.",
        "Avoid handling plants when wet.",
        "Use disease-free seeds."
    ],

    "early_blight": [
        "Remove infected leaves.",
        "Avoid overhead watering.",
        "Maintain proper spacing between plants."
    ],

    "healthy": [
        "Your tomato plant appears healthy!",
        "Continue regular watering.",
        "Provide sufficient sunlight and monitor the plant."
    ],

    "late_blight": [
        "Remove infected plant material.",
        "Avoid wetting the leaves.",
        "Improve air circulation around plants."
    ],

    "leaf_mold": [
        "Improve air circulation.",
        "Reduce humidity around the plant.",
        "Remove infected leaves."
    ],

    "mosaic_virus": [
        "Remove confirmed infected plants.",
        "Disinfect gardening tools.",
        "Use disease-free seeds."
    ],

    "septoria_leaf_spot": [
        "Remove infected lower leaves.",
        "Avoid overhead watering.",
        "Use mulch to reduce soil splash."
    ],

    "target_spot": [
        "Remove infected leaves.",
        "Improve plant spacing.",
        "Keep foliage dry."
    ],

    "twospotted_spider_mite": [
        "Check the undersides of leaves.",
        "Wash mites off with water for light infestations.",
        "Seek agricultural advice for severe infestations."
    ],

    "yellow_leaf_curl_virus": [
        "Remove confirmed infected plants.",
        "Control whiteflies.",
        "Use resistant tomato varieties where available."
    ]
}




if page == "AI Assistant":

    st.title("ðŸ¤– TomatoCare AI Assistant")
    st.caption("FAISS + Gemini Powered Disease Assistant")

    st.write(
        "Ask questions about tomato plant diseases, "
        "symptoms, prevention and treatment."
    )

    question = st.text_input(
        "Ask your question",
        placeholder="What are the symptoms of early blight?"
    )

    if st.button("ðŸ’¬ Ask TomatoCare AI"):

        if question.strip():

            with st.spinner("Generating answer..."):

                try:
                    answer = generate_answer(question)

                    st.subheader("ðŸ’¬ TomatoCare AI Answer")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Assistant error: {e}")

        else:
            st.warning("Please enter a question.")

    st.stop()


if page != "Disease Detection":
    st.stop()
plants_df = get_plants()

if not plants_df.empty:
    plant_options = {
        f"{row['plant_name']} (ID: {row['plant_id']})": row['plant_id']
        for _, row in plants_df.iterrows()
    }

    selected_plant = st.selectbox(
        "Select your registered tomato plant",
        options=list(plant_options.keys())
    )

    selected_plant_id = plant_options[selected_plant]

else:
    st.warning("Please register a plant before scanning.")
    selected_plant_id = None

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)
    
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    st.image(img, caption="Uploaded Tomato Leaf", width=300)

    if st.button("Detect Disease"):

        if selected_plant_id is None:
            st.warning("Please register or select a plant first.")

        else:
            with st.spinner("Detecting disease..."):

                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                prediction = model.predict(img_array, verbose=0)

                index = np.argmax(prediction[0])
                disease = categories[index]
                confidence = float(prediction[0][index]) * 100

                save_scan(
                    plant_id=selected_plant_id,
                    disease=disease,
                    confidence=confidence
                )

                st.success("Plant health scan saved successfully!")

                st.subheader("Detection Result")
                st.write("Detected Disease:", disease)
                st.write(f"Confidence: {confidence:.2f}%")

                st.subheader("Recommended Remedies")

                for remedy in remedies.get(
                    disease,
                    ["Consult a local agricultural expert."]
                ):
                    st.write("âœ…", remedy)

                st.warning(
                    "This is an AI-based prediction, not a confirmed diagnosis. "
                    "Verify serious or uncertain cases with an agricultural expert."
                )
                # Generate PDF report
                plants_df = get_plants()

                plant_match = plants_df[
                    plants_df["plant_id"] == selected_plant_id
                ]

                plant_name = (
                    plant_match.iloc[0]["plant_name"]
                    if not plant_match.empty
                    else "Tomato Plant"
                )

                scan_date = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                pdf_file = generate_pdf_report(
                    plant_name=plant_name,
                    plant_id=selected_plant_id,
                    disease=disease,
                    confidence=confidence,
                    scan_date=scan_date,
                    remedies=remedies.get(
                        disease,
                        ["Consult a local agricultural expert."]
                    )
                )

                st.download_button(
                    label="ðŸ“„ Download Disease Report (PDF)",
                    data=pdf_file,
                    file_name=f"TomatoCare_Report_{selected_plant_id}.pdf",
                    mime="application/pdf"
                )
