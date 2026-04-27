import streamlit as st
import pandas as pd
import joblib

# ==================== PAGE CONFIG & CUSTOM CSS ====================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #60a5fa !important;
        font-weight: 600;
    }
    .css-1d391kg, .stSelectbox, .stNumberInput {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #334155;
    }
    label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    .stSuccess, .stError {
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    .css-1v3fv5y {
        background-color: #0f172a;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="font-size: 3rem; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 Customer Churn Prediction
    </h1>
    <p style="color: #94a3b8; font-size: 1.2rem;">Predict customer retention using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR - MODEL SELECTION ====================
st.sidebar.markdown("<h2 style='color: #60a5fa;'>🔧 Model Selection</h2>", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose Prediction Model",
    options=[
        "Logistic Regression (Best Accuracy)",
        "Random Forest",
        "XGBoost"
    ],
    index=0
)

# Load model safely with correct filenames
with st.spinner(f"Loading {model_choice.split(' (')[0]} model..."):
    try:
        if model_choice == "Logistic Regression (Best Accuracy)":
            model = joblib.load("logistic_model.pkl")
            acc = "~80.3%"
        elif model_choice == "Random Forest":
            model = joblib.load("Random_forest.pkl")  # ← Exact match to your file!
            acc = "~78.4%"
        else:  # XGBoost
            model = joblib.load("xgb_model.pkl")
            acc = "~79.4%"
        
        st.sidebar.success(f"**{model_choice.split(' (')[0]}** loaded successfully!")
        st.sidebar.metric("Test Accuracy", acc)
    
    except Exception as e:
        st.sidebar.error("Model file not found!")
        st.sidebar.code(str(e))
        st.stop()

# ==================== MAIN INPUT FORM ====================
st.markdown("### Customer Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Select...", "Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["Select...", "No", "Yes"])
    Partner = st.selectbox("Partner", ["Select...", "Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Select...", "Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Select...", "Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Select...", "Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["Select...", "DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Select...", "Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Select...", "Yes", "No"])

st.markdown("### Add-on Services")
col3, col4 = st.columns(2)
with col3:
    OnlineSecurity = st.selectbox("Online Security", ["Select...", "Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Select...", "Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Select...", "Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Select...", "Yes", "No", "No internet service"])

with col4:
    StreamingTV = st.selectbox("Streaming TV", ["Select...", "Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Select...", "Yes", "No", "No internet service"])
    PaymentMethod = st.selectbox("Payment Method", ["Select...",
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

st.markdown("### Billing Details")
col5, col6 = st.columns(2)
with col5:
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.05)
with col6:
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=0.05)

# ==================== PREDICTION ====================
if st.button("🔮 Predict Customer Churn", use_container_width=True):
    required_fields = [
        gender, SeniorCitizen, Partner, Dependents, PhoneService,
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod
    ]

    if "Select..." in required_fields:
        st.warning("⚠️ Please complete all fields before predicting.")
    else:
        try:
            input_dict = {
                "gender": gender,
                "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": int(tenure),
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": float(MonthlyCharges),
                "TotalCharges": str(float(TotalCharges))
            }

            training_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]

            input_df = pd.DataFrame([input_dict])[training_columns]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            # FIX: Convert to normal float so XGBoost works with progress bar
            prob_churn = float(probability[1])     # Probability of Churn
            prob_stay  = float(probability[0])     # Probability of Stay

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction in ["Yes", 1]:
                st.error("### ⚠️ HIGH CHURN RISK")
                st.progress(prob_churn)
                st.markdown(f"**Churn Probability: {prob_churn:.1%}**")
                st.info("💡 **Recommendation**: Offer retention discount, upgrade plan, or loyalty program.")
            else:
                st.success("### ✅ LOW CHURN RISK")
                st.progress(prob_stay)
                st.markdown(f"**Retention Probability: {prob_stay:.1%}**")
                st.info("🎉 Great! This customer is likely to stay loyal.")

            st.caption(f"Prediction made using: **{model_choice.split(' (')[0]}**")

        except Exception as e:
            st.error("Prediction failed. Please try again.")
            st.exception(e)

# ==================== SOURCE CODE ====================
st.markdown("---")
with st.expander("📝 View Model Training Source Code", expanded=False):
    st.code("""
import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 


            # Load dataset            
df = pd.read_csv("customer_churn.csv")
            
df
            
    
    # Data Preprocessing        
df.drop(["customerID"] , inplace = True , axis =1)

df.duplicated()

df.isnull().sum()

df.dropna()
            
# Feature and Target Separation
            
x = df.drop("Churn" , axis =1 )
y = df["Churn"]
            

numeric_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = x.select_dtypes(include=["object", "bool"]).columns.tolist()
            
# Preprocessing Pipelines
            
preprocessor = ColumnTransformer(transformers = [
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
            
# Train-Test Split
            
x_train , x_test , y_train , y_test = train_test_split(
    x , y , random_state = 42 , stratify = y  , test_size = 0.2 )
            

           # Model Pipelines
            
linear_reg_model = Pipeline(steps = [

    ("preprocessor" , preprocessor),
    ("classifier" , LogisticRegression(max_iter= 1000))
])
            

Rfc_model = Pipeline(steps = [

    ( "preprocessor", preprocessor),
    ("classifier" , RandomForestClassifier(random_state = 42))

])

# Model Training and Evaluation      

linear_reg_model.fit(x_train ,y_train)

pred = linear_reg_model.predict(x_test)

pred

        # Accuracy Calculation
            

acc = accuracy_score(y_test, pred)
print("Linear_model:", acc)      
            

            # Random Forest Model Training

 Rfc_model.fit(x_train , y_train)
            
pred_2 =  Rfc_model.predict(x_test)
            
pred_2
            
acc = accuracy_score(y_test, pred_2)
print("Random_forest:", acc)
            

            # XGBoost Model Training

xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    ))
])
            
xgb_model.fit(x_train, y_train.map({"Yes": 1, "No": 0})
)

        # Prediction    
pred_3 = xgb_model.predict(x_test)
            
pred_3
            
acc = accuracy_score(y_test.map({"Yes": 1, "No": 0}), pred_3)
print("XGBoost Accuracy:", acc)



# Save Models
            
joblib.dump(linear_reg_model, "logistic_model.pkl")

print("succesfully downloaded ")
            

joblib.dump(Rfc_model, "Random_forest.pkl")

print("succesfully downloaded ")
            

joblib.dump(xgb_model, "xgb_model.pkl")

print("succesfully downloaded ")
            



                     

    """, language="text")
    st.info("All three models were trained and saved in your project folder. Logistic Regression performed best!")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🚀 Built with ❤️ using Streamlit • Logistic Regression achieved the highest accuracy</p>
    <p>Dataset: Telco Customer Churn</p>
</div>
""", unsafe_allow_html=True)