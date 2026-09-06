import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Purchase Intent Prediction",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for minor styling tweaks
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_model():
    # Make sure "online_shoppers_model.joblib" is in the same directory
    return joblib.load("online_shoppers_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading the model: {e}")
    st.info("Please ensure you have trained and saved the model as 'online_shoppers_model.joblib'.")
    st.stop()

# 4. Header Section
st.title("🛒 Purchase Intent Prediction System")
st.markdown("""
Welcome to the visitor analytics dashboard. Adjust the session parameters below 
to forecast whether a website visitor is likely to complete a transaction.
""")
st.divider()

# 5. Input Section (Using Tabs for better UX)
tab1, tab2, tab3 = st.tabs(["📊 Page Activity", "📈 Session Metrics", "👤 Visitor Profile"])

with tab1:
    st.markdown("#### Visitor Engagement by Page Type")
    col1_a, col1_b = st.columns(2)
    with col1_a:
        administrative = st.number_input("Administrative Pages", min_value=0, value=0, help="Number of account/management pages visited.")
        informational = st.number_input("Informational Pages", min_value=0, value=0, help="Number of informational/policy pages visited.")
        product_related = st.number_input("Product Related Pages", min_value=0, value=1, help="Number of product specific pages visited.")
    with col1_b:
        administrative_duration = st.number_input("Admin Duration (sec)", min_value=0.0, value=0.0)
        informational_duration = st.number_input("Info Duration (sec)", min_value=0.0, value=0.0)
        product_related_duration = st.number_input("Product Duration (sec)", min_value=0.0, value=10.0)

with tab2:
    st.markdown("#### Google Analytics Metrics")
    col2_a, col2_b = st.columns(2)
    with col2_a:
        bounce_rates = st.slider("Bounce Rate", 0.0, 1.0, 0.0, help="Percentage of visitors who enter the site and leave without viewing other pages.")
        exit_rates = st.slider("Exit Rate", 0.0, 1.0, 0.05, help="Percentage of pageviews that were the last in the session.")
    with col2_b:
        page_values = st.number_input("Page Value", min_value=0.0, value=0.0, help="Average value for a web page that a user visited before completing an e-commerce transaction.")
        special_day = st.slider("Special Day Closeness", 0.0, 1.0, 0.0, help="Closeness of the site visiting time to a specific special day (e.g., Mother's Day, Valentine's Day).")

with tab3:
    st.markdown("#### System & Demographics")
    col3_a, col3_b = st.columns(2)
    with col3_a:
        visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        region = st.number_input("Region ID", min_value=1, max_value=9, value=1)
    with col3_b:
        operating_systems = st.number_input("Operating System ID", min_value=1, max_value=8, value=1)
        browser = st.number_input("Browser ID", min_value=1, max_value=13, value=1)
        traffic_type = st.number_input("Traffic Type ID", min_value=1, max_value=20, value=1)

st.write("") # Spacer

# 6. Prediction Logic & Output
if st.button("🚀 Analyze Session & Predict", type="primary"):
    
    # Bundle inputs for the model
    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "Month": month,
        "VisitorType": visitor_type
    }])

    # Run inference
    with st.spinner("Analyzing session data..."):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
    
    # Display Results Professionally
    st.divider()
    st.subheader("📊 Analysis Results")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.metric(label="Forecast", value="Purchase Likely", delta="Positive Intent")
        else:
            st.metric(label="Forecast", value="No Purchase", delta="- Low Intent", delta_color="inverse")
            
    with res_col2:
        st.markdown("**Confidence Score:**")
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
        
        # Color code the progress bar based on the prediction
        if prediction == 1:
            st.progress(confidence, text=f"{confidence*100:.1f}% probability of converting")
            st.success("High likelihood of generating revenue from this session. Consider triggering a promotional popup to seal the deal.")
        else:
            st.progress(confidence, text=f"{confidence*100:.1f}% probability of bouncing/exiting")
            st.warning("User is exhibiting browsing behavior with low purchase intent.")