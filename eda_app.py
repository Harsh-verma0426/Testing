import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, contextlib
from data_cleaning import data_clean
from auto_eda import EDA

st.set_page_config(page_title="Automated EDA Tool", layout="wide")
st.title("Automated EDA & Data Cleaning App")

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

    st.success(f"Successfully loaded {uploaded_file.name}")

    # --- Preserve dataframe between reruns ---
    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    df = st.session_state.df

    # --- Data Overview ---
    with st.expander("Data Overview"):
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
            df = data_clean.remove_duplicates(df)
        st.text(log_stream.getvalue())
        st.success("Duplicates removed successfully!")

    # --- Data Type Conversion ---
    if st.button("Convert Data Types Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = data_clean.data_type_conversion(df)
        st.text(log_stream.getvalue())
        st.success("Data type conversion complete!")

    # --- Missing Value Summary ---
    st.write("###Missing Value Summary")
    null_counts = df.isnull().sum()
    st.dataframe(null_counts[null_counts > 0])

    # --- Fill Missing Values ---
    if st.button("Fill Missing Values Automatically"):
        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            df = data_clean.fill_missing_values(df)
        st.text_area("Cleaning Log", log_stream.getvalue(), height=300)
        st.success("Missing values filled successfully!")

    # --- Cleaned Output ---
    st.write("### Cleaned Data Sample")
    st.dataframe(df.head(20))

    # --- Data Overview ---
    with st.expander("📊 Data Overview"):
        st.subheader("Dataset Information")

        info_df = EDA.get_data_overview(df)

        # Highlight missing values visually
        def highlight_missing(val):
            return 'background-color: #ffcccc' if val < len(df) else ''

        styled_info = info_df.style.applymap(highlight_missing, subset=['Non-Null Count'])
        st.dataframe(styled_info, use_container_width=True)

        # Statistical summary
        st.subheader("Statistical Summary")
        stats_df = EDA.get_statistical_summary(df)
        st.dataframe(stats_df, use_container_width=True)

        # Sample rows
        st.subheader("Sample Rows")
        st.dataframe(df.head(), use_container_width=True)
        st.success("Data overview generated successfully!")

    # --- Correlation Heatmap ---
    st.write("### 🔗 Correlation Heatmap")
    with st.expander("Show Heatmap"):
        corr_matrix = EDA.correlation_matrix(df)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        st.pyplot(fig)

    # --- Exploratory Data Analysis Section ---
    st.write("## 🔍 Exploratory Data Analysis")

    useful_cols = EDA.get_useful_columns(df)
    if not useful_cols:
        st.warning("⚠️ No suitable columns found for visualization.")
    else:
        type_choice = st.radio("Select column type to explore:", ["Numeric", "Categorical", "Datetime"])
        filtered_cols = EDA.get_columns_by_type(df, type_choice, useful_cols)

        if not filtered_cols:
            st.warning(f"⚠️ No {type_choice.lower()} columns found in this dataset.")
        else:
            col = st.selectbox(f"Select a {type_choice.lower()} column to visualize", filtered_cols)
            sample_df = EDA.sample_dataframe(df)

            # --- Numeric columns ---
            if type_choice == "Numeric":
                st.write(f"### 📊 Distribution of `{col}` (sample of {len(sample_df)} rows)")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(sample_df[col], bins=30, kde=True, ax=ax, color="skyblue")
                st.pyplot(fig)

                st.write("### 🧮 Boxplot (to see outliers)")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.boxplot(x=sample_df[col], color="orange", ax=ax)
                st.pyplot(fig)

            # --- Categorical columns ---
            elif type_choice == "Categorical":
                st.write(f"### 📊 Frequency of top categories in `{col}`")
                counts = sample_df[col].value_counts().head(20)
                st.bar_chart(counts)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                st.write(f"### 🗓️ Trend over time for `{col}`")

                # Work on a copy — never modify df in place
                temp_col = pd.to_datetime(df[col].copy(), errors='coerce', dayfirst=True)
                date_counts = temp_col.value_counts().sort_index()

                trend_df = date_counts.reset_index()
                trend_df.columns = [col, "Count"]

                st.line_chart(trend_df, x=col, y="Count")

                # Optional: check dtype
                st.caption(f"Column `{col}` dtype: {df[col].dtype}")

            else:
                st.warning("⚠️ This column type is not supported for visualization yet.")

    with st.expander("🧬 Data Types Overview"):
        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Dtype": [df[c].dtype for c in df.columns]
        }))

    st.write("## 🔗 Two-Column Relationship Analysis")

    # Filter columns first
    useful_cols = EDA.get_useful_columns(df)

    if not useful_cols:
        st.warning("⚠️ No suitable columns available for relationship plotting.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select first column (X-axis)", useful_cols, key="x_col")
        with col2:
            y_col = st.selectbox("Select second column (Y-axis)", useful_cols, key="y_col")

        if x_col and y_col:
            result = EDA.prepare_bivariate_data(df, x_col, y_col)

            if result["plot_type"] == "scatter":
                fig, ax = plt.subplots()
                sns.scatterplot(data=result["data"], x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif result["plot_type"] == "bar":
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=result["data"], x=x_col, y=f"mean_{y_col}", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            elif result["plot_type"] == "line":
                fig, ax = plt.subplots()
                sns.lineplot(data=result["data"], x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif result["plot_type"] == "grouped_bar":
                st.bar_chart(result["data"])

            else:
                st.warning("⚠️ This column combination is not supported yet.")

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

