import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
import io

st.set_page_config(page_title="SVM Learning Hub", page_icon="??", layout="wide")

st.title("?? SVM Learning Hub")
st.markdown("Learn and implement Support Vector Machines with Python.")

tab1, tab2, tab3, tab4 = st.tabs(["?? Learning", "?? Interactive Lab", "?? CSV Training", "? Quiz"])

with tab1:
    st.header("SVM Fundamentals")
    st.markdown("""
    **Support Vector Machine (SVM)** is a supervised learning algorithm used for classification and regression.
    
    **Key Concepts:**
    - **Hyperplane**: Decision boundary that separates classes
    - **Support Vectors**: Data points closest to the hyperplane
    - **Margin**: Distance between hyperplane and nearest support vectors
    - **Kernel Trick**: Maps data to higher dimensions for non-linear separation
    
    **Kernels:** Linear, Polynomial, RBF (Radial Basis Function), Sigmoid
    """)
    
    st.subheader("Python Example")
    st.code("""
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)
print("Accuracy:", clf.score(X, y))
    """, language="python")

with tab2:
    st.header("Interactive SVM Lab")
    dataset = st.selectbox("Dataset", ["Blobs", "Moons", "Circles"])
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("C (regularization)", 0.01, 100.0, 1.0, log=True)
    gamma = st.slider("Gamma", 0.001, 10.0, 1.0, log=True)
    degree = st.slider("Degree (poly)", 2, 5, 3)
    
    if st.button("Train Model"):
        if dataset == "Blobs":
            X, y = make_blobs(n_samples=200, centers=2, random_state=42)
        elif dataset == "Moons":
            X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
        else:
            X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)
        
        if kernel == "poly":
            clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        else:
            clf = SVC(kernel=kernel, C=C, gamma=gamma)
        clf.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=30)
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='green', linewidths=2, label='Support Vectors')
        ax.set_title(f"SVM ({kernel}) - Accuracy: {clf.score(X, y):.3f}")
        ax.legend()
        st.pyplot(fig)
        st.write(f"**Support Vectors:** {len(clf.support_vectors_)}")
        st.write(f"**Accuracy:** {clf.score(X, y):.4f}")

with tab3:
    st.header("CSV Dataset Training")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head(10))
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        
        target = st.selectbox("Target column", df.columns)
        features = st.multiselect("Feature columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target][:2])
        
        if features and st.button("Train SVM"):
            X = df[features].values
            y = df[target].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            st.write(f"**Test Accuracy:** {accuracy_score(y_test, pred):.4f}")
            st.write("**Confusion Matrix:**")
            st.write(confusion_matrix(y_test, pred))
            st.write("**Classification Report:**")
            st.text(classification_report(y_test, pred))

with tab4:
    st.header("Quiz")
    questions = [
        {"q": "What does SVM stand for?", "opts": ["Support Vector Machine", "Simple Vector Model", "Statistical Variance Method", "Super Vector Math"], "ans": 0},
        {"q": "Which kernel is best for non-linear data?", "opts": ["Linear", "RBF", "Sigmoid", "None"], "ans": 1},
        {"q": "What is a support vector?", "opts": ["Any data point", "Points closest to hyperplane", "Outliers", "Centroids"], "ans": 1},
    ]
    score = 0
    for i, q in enumerate(questions):
        choice = st.radio(q["q"], q["opts"], key=f"q{i}")
        if choice == q["opts"][q["ans"]]:
            score += 1
    st.write(f"**Score: {score}/{len(questions)}**")

st.sidebar.markdown("### SVM Learning Hub")
st.sidebar.markdown("Built with Streamlit + scikit-learn")