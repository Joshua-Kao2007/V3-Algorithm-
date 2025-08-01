import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Setup ---
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📁 ML Training & Evaluation Dashboard")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🧪 Behavioral/iWave Modeling", "📈 Stacked Classifier Summary", "🧠 Predict New Data"])

# --- TAB 1 ---
with tab1:
    # --- Upload Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        patron_behavioral = st.file_uploader("📄 Patron-Cleaned Behavioral", type="csv", key="pb")
        patron_iwave = st.file_uploader("📄 Patron-Cleaned iWave", type="csv", key="pi")
    with col2:
        under_behavioral = st.file_uploader("📄 Under-Cleaned Behavioral", type="csv", key="ub")
        under_iwave = st.file_uploader("📄 Under-Cleaned iWave", type="csv", key="ui")

    st.subheader("🧭 Choose Dataset Type")
    data_type = st.radio("Which dataset do you want to use?", ["Behavioral", "iWave"])

    def load_and_tag(file, label):
        try:
            df = pd.read_csv(file)
            df["Category"] = label
            return df
        except:
            return None

    dfs = []
    if data_type == "Behavioral":
        if patron_behavioral:
            dfs.append(load_and_tag(patron_behavioral, "Patron"))
        if under_behavioral:
            dfs.append(load_and_tag(under_behavioral, "Under"))
        output_file = "Combined_Behavioral.xlsx"
    elif data_type == "iWave":
        if patron_iwave:
            dfs.append(load_and_tag(patron_iwave, "Patron"))
        if under_iwave:
            dfs.append(load_and_tag(under_iwave, "Under"))
        output_file = "Combined_iWave.xlsx"

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Combined {len(dfs)} file(s)")
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head(10))

        df.to_excel(output_file, index=False)
        st.download_button("⬇️ Download Combined Excel", data=open(output_file, "rb").read(),
                        file_name=output_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("🧠 Model Training")
        if st.button("Train Models on This Dataset"):
            try:
                if "Category" not in df.columns:
                    st.error("❌ No 'Category' column found.")
                    st.stop()

                le = LabelEncoder()
                df["Target"] = le.fit_transform(df["Category"])
                class_labels = le.classes_

                X = df.select_dtypes(include='number').drop(columns=["Target"], errors="ignore").fillna(0)
                y = df["Target"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs'),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "AdaBoost": AdaBoostClassifier()
                }

                metrics = []
                plots = {}

                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    proba = model.predict_proba(X_test_scaled)

                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    auc = roc_auc_score(y_test, proba[:, 1]) if len(class_labels) == 2 else \
                        roc_auc_score(y_test, proba, multi_class='ovr', average='macro')

                    metrics.append({"Model": name, "Accuracy (%)": round(acc * 100, 2),
                                    "F1 Score": round(f1, 2), "AUC": round(auc, 2)})

                    plots[name] = {"confusion": confusion_matrix(y_test, preds), "fpr": None, "tpr": None}

                    if len(class_labels) == 2:
                        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                        plots[name]["fpr"] = fpr
                        plots[name]["tpr"] = tpr

                st.subheader("📊 Model Evaluation")
                st.dataframe(pd.DataFrame(metrics))

                for name, p in plots.items():
                    st.markdown(f"### 🔍 {name}")

                    fig1, ax1 = plt.subplots()
                    sns.heatmap(p["confusion"], annot=True, fmt="d", cmap="Blues",
                                xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
                    ax1.set_xlabel("Predicted")
                    ax1.set_ylabel("Actual")
                    ax1.set_title("Confusion Matrix")
                    st.pyplot(fig1)

                    if p["fpr"] is not None:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(p["fpr"], p["tpr"], label="ROC Curve")
                        ax2.plot([0, 1], [0, 1], 'k--')
                        ax2.set_title("ROC Curve (Binary Only)")
                        ax2.set_xlabel("False Positive Rate")
                        ax2.set_ylabel("True Positive Rate")
                        ax2.legend()
                        st.pyplot(fig2)

                # Save for use in tab 2
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['scaler'] = scaler
                st.session_state['class_labels'] = class_labels

                # Optionally save model dict for future use
                st.session_state['stack_ready'] = True

            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
    else:
        st.info("👆 Upload at least one Patron and one Under file to begin.")

# --- TAB 2 ---
with tab2:
    st.subheader("📈 Summary Page — Stacked Ensemble Model")
    if st.session_state.get("stack_ready"):
        X = st.session_state['X']
        y = st.session_state['y']
        scaler = st.session_state['scaler']
        class_labels = st.session_state['class_labels']

        X_scaled = scaler.transform(X)

        base_estimators = [
            ("lr", LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')),
            ("knn", KNeighborsClassifier()),
            ("rf", RandomForestClassifier()),
            ("ada", AdaBoostClassifier())
        ]

        stack_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            passthrough=True
        )

        try:
            stack_model.fit(X_scaled, y)
            preds = stack_model.predict(X_scaled)
            proba = stack_model.predict_proba(X_scaled)

            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="weighted")
            auc = roc_auc_score(y, proba[:, 1]) if len(class_labels) == 2 else \
                roc_auc_score(y, proba, multi_class='ovr', average='macro')

            st.success("✅ Stacked model trained and evaluated on full dataset.")
            st.dataframe(pd.DataFrame([{
                "Model": "Stacked Ensemble",
                "Accuracy (%)": round(acc * 100, 2),
                "F1 Score": round(f1, 2),
                "AUC": round(auc, 2)
            }]))

            fig_cm, ax_cm = plt.subplots()
            cm = confusion_matrix(y, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax_cm,
                        xticklabels=class_labels, yticklabels=class_labels)
            ax_cm.set_title("Confusion Matrix — Stacked Classifier")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            if len(class_labels) == 2:
                fpr, tpr, _ = roc_curve(y, proba[:, 1])
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label="Stacked ROC")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_title("ROC Curve — Stacked (Binary Only)")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                st.pyplot(fig_roc)

        except Exception as e:
            st.error(f"❌ Failed to train Stacking Classifier: {e}")
    else:
        st.warning("Train a model in the first tab before viewing the summary.")