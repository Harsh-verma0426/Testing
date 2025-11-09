import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, contextlib
from Automated_EDA import EDA

st.set_page_config(page_title="Automated EDA Tool", layout="wide")
st.title("🤖 Automated EDA & Data Cleaning App")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    elif file_type == "tsv":
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.success(f"✅ Successfully loaded {uploaded_file.name}")

    # --- Data Overview ---
    with st.expander("📊 Data Overview"):
        st.subheader("Dataset Information")

        # Create DataFrame similar to df.info() output
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [df[col].count() for col in df.columns],
            "Dtype": [df[col].dtype for col in df.columns]
        })

        # Highlight missing values
        def highlight_missing(val):
            return 'background-color: #ffcccc' if val < len(df) else ''

        styled_info = info_df.style.applymap(highlight_missing, subset=['Non-Null Count'])
        st.dataframe(styled_info, use_container_width=True)

        # Show basic stats
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        # Optional: Show sample rows
        st.subheader("Sample Rows")
        st.dataframe(df.head(), use_container_width=True)

    # --- Remove Duplicates ---
    if st.button("Remove Duplicates"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.remove_duplicates(df)
        st.text(log_stream.getvalue())
        st.success("Duplicates removed successfully!")

    # --- Data Type Conversion ---
    if st.button("Convert Data Types Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.data_type_conversion(df)
        st.text(log_stream.getvalue())
        st.success("✅ Data type conversion complete!")

    # --- Missing Value Summary ---
    st.write("### ⚠️ Missing Value Summary")
    null_counts = df.isnull().sum()
    st.dataframe(null_counts[null_counts > 0])

    # --- Fill Missing Values ---
    if st.button("Fill Missing Values Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = EDA.fill_missing_values(df)
        st.text_area("🧾 Cleaning Log", log_stream.getvalue(), height=300)
        st.success("✅ Missing values filled successfully!")

    # --- Cleaned Output ---
    st.write("### 🧾 Cleaned Data Sample")
    st.dataframe(df.head(20))

    # --- Correlation Heatmap ---
    st.write("### 📈 Correlation Heatmap")
    corr_matrix = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    st.pyplot(fig)

    # --- Download ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Please upload a CSV, Excel, or TSV file to begin.")
