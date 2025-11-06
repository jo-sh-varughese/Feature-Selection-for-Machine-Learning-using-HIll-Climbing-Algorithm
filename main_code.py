import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_diabetes, fetch_california_housing
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 🧠 Hill Climbing Feature Selection Algorithm
# -----------------------------------------------------------
def hill_climbing_feature_selection(X, y, model):
    n_features = X.shape[1]
    current_features = set(random.sample(range(n_features), min(3, n_features)))
    best_score = np.mean(cross_val_score(model, X.iloc[:, list(current_features)], y, cv=5))
    history = [(len(current_features), best_score)]
    improved = True

    while improved:
        improved = False
        for i in range(n_features):
            neighbor = current_features.copy()
            if i in current_features:
                neighbor.remove(i)
            else:
                neighbor.add(i)
            if len(neighbor) == 0:
                continue
            score = np.mean(cross_val_score(model, X.iloc[:, list(neighbor)], y, cv=5))
            if score > best_score:
                best_score = score
                current_features = neighbor
                history.append((len(current_features), best_score))
                improved = True
                break

    selected_features = X.columns[list(current_features)]
    return selected_features, best_score, history


# -----------------------------------------------------------
# ⚙️ Streamlit Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="AI Feature Selector", page_icon="🧠", layout="wide")

st.title("🧠 AI Feature Selector — Hill Climbing Optimization")
st.markdown("""
A professional machine learning web app that automatically selects the **most important features**  
using the **Hill Climbing optimization algorithm** — supporting both **classification and regression** tasks.
""")

# -----------------------------------------------------------
# 📂 Sidebar Setup
# -----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Dataset selection
    dataset_option = st.selectbox(
        "📊 Choose Dataset",
        ["Upload your own CSV", "Breast Cancer", "Wine", "Iris", "Diabetes", "California Housing"]
    )

    uploaded_file = None
    if dataset_option == "Upload your own CSV":
        uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=["csv"])

    model_choice = st.selectbox(
        "🤖 Choose ML Model",
        ["Decision Tree", "Random Forest", "Logistic Regression / Linear Regression"]
    )

    st.markdown("---")
    st.info("💡 Tip: Use built-in datasets for quick demonstration to judges or interviewers!")

# -----------------------------------------------------------
# 📥 Load Dataset (either from sklearn or user-uploaded)
# -----------------------------------------------------------
def load_builtin_dataset(name):
    if name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        return data.frame.drop(columns=["target"]), data.target, "classification"
    elif name == "Wine":
        data = load_wine(as_frame=True)
        return data.frame.drop(columns=["target"]), data.target, "classification"
    elif name == "Iris":
        data = load_iris(as_frame=True)
        return data.frame.drop(columns=["target"]), data.target, "classification"
    elif name == "Diabetes":
        data = load_diabetes(as_frame=True)
        return data.data, data.target, "regression"
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        return data.frame.drop(columns=["MedHouseVal"]), data.target, "regression"
    else:
        return None, None, None

if dataset_option != "Upload your own CSV":
    X, y, task_type = load_builtin_dataset(dataset_option)
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        task_type = "classification" if y.nunique() <= 20 else "regression"
    else:
        st.stop()

# -----------------------------------------------------------
# 🔧 Encode Data
# -----------------------------------------------------------
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

if task_type == "classification" and y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

# -----------------------------------------------------------
# ⚙️ Model Selection
# -----------------------------------------------------------
if model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42) if task_type == "classification" else DecisionTreeRegressor(random_state=42)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
else:
    model = LogisticRegression(max_iter=2000) if task_type == "classification" else LinearRegression()

# -----------------------------------------------------------
# 🧠 Run Hill Climbing Optimization
# -----------------------------------------------------------
st.subheader("🚀 Running Hill Climbing Feature Selection...")
with st.spinner("Optimizing feature subset..."):
    best_features, best_accuracy, history = hill_climbing_feature_selection(X, y, model)

st.success("✅ Optimization Completed Successfully!")
st.write(f"**Best Cross-Validation Score:** `{best_accuracy:.4f}`")
st.write(f"**Selected Features ({len(best_features)}):** {list(best_features)}")

# -----------------------------------------------------------
# 📈 Plotly: Optimization Progress
# -----------------------------------------------------------
hist_df = pd.DataFrame(history, columns=["Number of Features", "Score"])
fig_progress = px.line(
    hist_df,
    x="Number of Features",
    y="Score",
    markers=True,
    title="📈 Optimization Path (Score vs Features)",
    color_discrete_sequence=["#636EFA"]
)
fig_progress.update_traces(line=dict(width=3))
st.plotly_chart(fig_progress, use_container_width=True)

# -----------------------------------------------------------
# 🧪 Train-Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X[best_features], y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------------------
# 📊 Model Evaluation
# -----------------------------------------------------------
st.subheader("📊 Model Performance Report")

if task_type == "classification":
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    st.metric("Training Accuracy", f"{train_acc:.4f}")
    st.metric("Testing Accuracy", f"{test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error", f"{mse:.4f}")
    st.metric("R² Score", f"{r2:.4f}")

    # Scatter Plot for Predictions
    fig_pred = px.scatter(
        x=y_test, y=y_pred,
        title="Actual vs Predicted Values",
        labels={'x': 'Actual', 'y': 'Predicted'},
        color_discrete_sequence=["#EF553B"]
    )
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="green", dash="dot")))
    st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------------------------------------
# 🔥 Feature Importance (Interactive)
# -----------------------------------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("🔥 Feature Importance (Interactive)")
    importance = pd.DataFrame({
        "Feature": best_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance,
        x="Feature",
        y="Importance",
        title="Feature Importance Ranking",
        color="Importance",
        color_continuous_scale="blues"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------
# 🌳 Decision Tree Visualization (if applicable)
# -----------------------------------------------------------
if model_choice == "Decision Tree":
    st.subheader("🌳 Decision Tree Structure")
    fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
    plot_tree(model, filled=True, feature_names=best_features, rounded=True)
    st.pyplot(fig_tree)

# -----------------------------------------------------------
# 📥 Download Report
# -----------------------------------------------------------
st.subheader("📄 Downloadable Report")
report = io.StringIO()
report.write("AI Feature Selector Report\n")
report.write("==========================\n")
report.write(f"Dataset: {dataset_option}\n")
report.write(f"Task Type: {task_type}\n")
report.write(f"Model: {model_choice}\n")
report.write(f"Best Cross-Validation Score: {best_accuracy:.4f}\n")
if task_type == "classification":
    report.write(f"Test Accuracy: {test_acc:.4f}\n")
else:
    report.write(f"R² Score: {r2:.4f}\n")
report.write("\nSelected Features:\n")
for feat in best_features:
    report.write(f"- {feat}\n")

st.download_button(
    label="📥 Download Full Report",
    data=report.getvalue(),
    file_name="AI_Feature_Selector_Report.txt"
)
