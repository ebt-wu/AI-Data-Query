
import os
import re
import json
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv
from db import init_db, save_prompt, get_all_prompts, clear_all_prompts


def build_prompt_from_inputs(operation, column=None, operator=None, value=None, limit=None,
                               custom_query=None, agg_function=None, group_by=None, sort_order=None):
    if operation == "Filter":
        query_json = {
            "operation": "filter",
            "column": column,
            "operator": operator,
            "value": value,
            "limit": limit
        }
        if operator == "contains":
            prompt = (f"Using pandas, filter the DataFrame for rows where the '{column}' column "
                      f"contains '{value}' using df['{column}'].str.contains('{value}', case=False, na=False). "
                      f"Return the first {limit} rows of the filtered DataFrame.")
        else:  
            prompt = (f"Using pandas, filter the DataFrame for rows where the '{column}' column equals '{value}'. "
                      f"Return the first {limit} rows of the filtered DataFrame.")
        return query_json, prompt

    elif operation == "Find Row":
        query_json = {
            "operation": "find_row",
            "column": column,
            "operator": "equals",
            "value": value,
            "limit": limit
        }
        prompt = (f"Using pandas, find rows where the '{column}' column equals '{value}'. "
                  f"Return the first {limit} rows of the result.")
        return query_json, prompt

    elif operation == "Aggregate":
        query_json = {
            "operation": "aggregate",
            "column": column,
            "function": agg_function,
            "group_by": group_by if group_by != "None" else None,
            "limit": limit
        }
        if agg_function.lower() == "describe":
            prompt = (f"Using pandas, generate summary statistics for the '{column}' column by calling "
                      f"df['{column}'].describe().")
        else:
            prompt = f"Using pandas, compute the {agg_function} of the '{column}' column"
            if group_by and group_by != "None":
                prompt += f" grouped by the '{group_by}' column"
            prompt += "."
        return query_json, prompt

    elif operation == "Sort":
        query_json = {
            "operation": "sort",
            "column": column,
            "order": sort_order,
            "limit": limit
        }
        prompt = (f"Using pandas, sort the DataFrame by the '{column}' column in {sort_order} order. "
                  f"Return the first {limit} rows of the sorted DataFrame.")
        return query_json, prompt

    elif operation == "Custom Query":
        query_json = {
            "operation": "custom",
            "query": custom_query
        }
        prompt = custom_query
        return query_json, prompt

    else:
        query_json = {
            "operation": "custom",
            "query": custom_query
        }
        prompt = custom_query
        return query_json, prompt

def format_raw_input(operation, column, operator, value, limit, sort_order, custom_query):
   
    if operation == "Custom Query":
        return custom_query
    elif operation in ["Filter", "Find Row"]:
        return (f"Operation: {operation}\nColumn: {column}\nOperator: {operator}\n"
                f"Value: {value}\nLimit: {limit}")
    elif operation == "Sort":
        return f"Operation: {operation}\nColumn: {column}\nOrder: {sort_order}\nLimit: {limit}"
    else:
        return ""



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_token=api_key)

st.set_page_config(page_title="Edbert", layout="wide")
st.markdown(
    """
    <style>
    .header-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header-subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="header-title">Edbert</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Clean and Analyze Your Files</div>', unsafe_allow_html=True)

init_db()


if "custom_query_input" not in st.session_state:
    st.session_state.custom_query_input = ""

if "custom_query_version" not in st.session_state:
    st.session_state.custom_query_version = 0

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []


uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)
max_file_size = 5 * 1024 * 1024
valid_files = []
if uploaded_files:
    for file in uploaded_files:
        if file.size > max_file_size:
            st.error(f"File {file.name} exceeds max file size of 5MB and was skipped.")
        else:
            valid_files.append(file)

if valid_files:
    file_names = [f.name for f in valid_files]
    selected_file_name = st.selectbox("Select a file", file_names)
    selected_file = next(f for f in valid_files if f.name == selected_file_name)
    
    try:
        if selected_file_name.endswith(".csv"):
            df_raw = pd.read_csv(selected_file, header=None)
        else:
            xls = pd.ExcelFile(selected_file)
            sheet_name = st.selectbox("Select a sheet", xls.sheet_names)
            df_raw = xls.parse(sheet_name, header=None)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df_raw = None

    if df_raw is not None:
        df_raw = df_raw.dropna(axis=0, how='all').dropna(axis=1, how='all').reset_index(drop=True)
        
        new_header = df_raw.iloc[0]
        if new_header.isnull().all() or new_header.duplicated().any():
            new_header = [f"column_{i}" for i in range(1, len(df_raw.columns) + 1)]
        df_raw = df_raw[1:]
        df_raw.columns = new_header

        def clean_data(df, remove_partial_rows):
            if df is None or df.empty:
                return None
            df.dropna(how="all", inplace=True)
            df.drop_duplicates(inplace=True)
            for col in df.columns:
                try:
                    col_series = df[col]
                except Exception as e:
                    st.error(f"Error accessing column {col}: {e}")
                    continue
                if pd.api.types.is_object_dtype(col_series):
                    df[col] = col_series.fillna("Unknown")
                elif pd.api.types.is_numeric_dtype(col_series):
                    df[col] = col_series.fillna(col_series.median())
                elif pd.api.types.is_datetime64_any_dtype(col_series):
                    df[col] = col_series.fillna(method="ffill")
            df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
            def safe_format(x):
                if isinstance(x, str):
                    return x.strip().title()
                return x
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(safe_format)
            for col in df.select_dtypes(include=["object"]).columns:
                s = pd.to_numeric(df[col].replace("Unknown", pd.NA), errors='coerce')
                if s.notna().sum() > 0.5 * len(s):
                    df[col] = s
            if remove_partial_rows:
                df.dropna(inplace=True)
            return df
        
        df_cleaned = clean_data(df_raw, remove_partial_rows=False)
        
        display_all = st.checkbox("Display all rows", value=False)
        default_n_rows = len(df_cleaned) if display_all else (5 if len(df_cleaned) >= 5 else len(df_cleaned))
        n_rows = st.number_input("Show top N rows:", min_value=1, max_value=len(df_cleaned), value=default_n_rows)
        
        st.subheader("Data Preview")
        tab1, tab2 = st.tabs(["Raw Data", "Cleaned Data"])
        with tab1:
            st.write("Raw Data (Before Cleaning)")
            st.dataframe(df_raw if display_all else df_raw.head(n_rows))
        with tab2:
            st.write("Cleaned Data (After Processing)")
            st.dataframe(df_cleaned if display_all else df_cleaned.head(n_rows))
        
        st.markdown("---")
    
        st.subheader("Statistical Analysis")
        numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stat_column = st.selectbox("Select Column for Analysis", options=numeric_cols)
            statistic = st.selectbox("Select Statistic", options=["mean", "std", "median", "min", "max", "count", "describe"])
            if st.button("Compute Statistic"):
                if statistic == "mean":
                    result = df_cleaned[stat_column].mean()
                elif statistic == "std":
                    result = df_cleaned[stat_column].std()
                elif statistic == "median":
                    result = df_cleaned[stat_column].median()
                elif statistic == "min":
                    result = df_cleaned[stat_column].min()
                elif statistic == "max":
                    result = df_cleaned[stat_column].max()
                elif statistic == "count":
                    result = df_cleaned[stat_column].count()
                elif statistic == "describe":
                    result = df_cleaned[stat_column].describe()
                st.write("Result:")
                st.write(result)
        else:
            st.warning("No numeric columns available for statistical analysis.")

        st.subheader("Data Visualization")

        
        chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot", "Box Plot"], key="chart_type")

        if chart_type == "Histogram":
            num_col = st.selectbox("Select Column for Histogram", options=df_cleaned.select_dtypes(include=["number"]).columns.tolist(), key="hist_column")
        else:
            x_axis = st.selectbox("Select X-axis", options=df_cleaned.columns.tolist(), key="viz_x_axis")
            y_axis = st.selectbox("Select Y-axis", options=df_cleaned.select_dtypes(include=["number"]).columns.tolist(), key="viz_y_axis")

        if st.button("Generate Chart"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            if chart_type == "Line Chart":
                
                sorted_df = df_cleaned.sort_values(by=x_axis)
                ax.plot(sorted_df[x_axis], sorted_df[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Line Chart of {y_axis} vs {x_axis}")
            elif chart_type == "Bar Chart":
                grouped = df_cleaned.groupby(x_axis)[y_axis].mean().reset_index()
                ax.bar(grouped[x_axis], grouped[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(f"Average {y_axis}")
                ax.set_title(f"Bar Chart of {y_axis} by {x_axis}")
            elif chart_type == "Histogram":
                ax.hist(df_cleaned[num_col], bins=20)
                ax.set_xlabel(num_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {num_col}")
            elif chart_type == "Scatter Plot":
                ax.scatter(df_cleaned[x_axis], df_cleaned[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Scatter Plot of {y_axis} vs {x_axis}")
            elif chart_type == "Box Plot":
                ax.boxplot(df_cleaned[y_axis])
                ax.set_xlabel(y_axis)
                ax.set_title(f"Box Plot of {y_axis}")
            st.pyplot(fig)

        st.subheader("Ask me anything about the data")
        ops = ["Filter", "Find Row", "Sort", "Custom Query"]
        default_op = st.session_state.get("desired_operation", "Filter")
        if default_op not in ops:
            default_op = "Filter"
        op_index = ops.index(default_op)
        operation = st.selectbox("Select Operation", ops, index=op_index, key="operation")
        
        column = operator = value = limit = sort_order = custom_query = None
        if operation in ["Filter", "Find Row"]:
            column = st.selectbox("Select Column", options=df_cleaned.columns.tolist(), key="ai_column")
            if operation == "Filter":
                operator = st.selectbox("Select Operator", options=["contains", "equals"], key="ai_operator")
                value = st.text_input("Enter value to filter by", key="ai_value")
                limit = st.number_input("Return first N rows", min_value=1, value=20, key="ai_limit")
            else:
                operator = "equals"
                value = st.text_input("Enter value to find", key="ai_value")
                limit = st.number_input("Return first N rows", min_value=1, value=1, key="ai_limit")
        elif operation == "Sort":
            column = st.selectbox("Select Column to Sort", options=df_cleaned.columns.tolist(), key="ai_column")
            sort_order = st.selectbox("Select Order", options=["ascending", "descending"], key="ai_order")
            limit = st.number_input("Return first N rows", min_value=1, value=20, key="ai_limit")
        elif operation == "Custom Query":
            
            custom_query = st.text_area("Enter your custom query", 
                                        value=st.session_state.get("custom_query_input", ""), 
                                        key=f"custom_query_input_{st.session_state.custom_query_version}")
        
        display_all_query = st.checkbox("Display all rows in query result", value=False, key="display_all_query")
        if display_all_query:
            limit = len(df_cleaned)
        
        if st.button("Submit Query"):
            query_json, final_prompt = build_prompt_from_inputs(operation, column, operator, value, limit,
                                                                custom_query, None, None, sort_order)
            st.info(f"Structured Query (JSON): {json.dumps(query_json, indent=2)}")
            st.info(f"Final Prompt: {final_prompt}")
            raw_input_text = ""
            if operation == "Custom Query":
                raw_input_text = custom_query
            elif operation in ["Filter", "Find Row"]:
                raw_input_text = (f"Operation: {operation}\nColumn: {column}\nOperator: {operator}\n"
                                  f"Value: {value}\nLimit: {limit}")
            elif operation == "Sort":
                raw_input_text = f"Operation: {operation}\nColumn: {column}\nOrder: {sort_order}\nLimit: {limit}"
            else:
                raw_input_text = final_prompt
            st.session_state.prompt_history.append((selected_file_name, raw_input_text))
            save_prompt(raw_input_text)
            
            smart_df = SmartDataframe(df_cleaned, config={"llm": llm})
            with st.spinner("Loading AI response, please wait..."):
                try:
                    response = smart_df.chat(final_prompt)
                except Exception as e:
                    st.error(f"Error: {e}")
                    response = None
                if response is not None:
                    if isinstance(response, pd.DataFrame):
                        if response.empty:
                            st.warning("No answer was returned by the AI. Please try a different query or check your data.")
                        else:
                            st.success("Answer:")
                            st.dataframe(response)
                    else:
                        if not response or str(response).strip() == "":
                            st.warning("No answer was returned by the AI. Please try a different query or check your data.")
                        else:
                            st.success("Answer:")
                            st.write(response)
                    feedback = st.radio("Was this answer helpful?", ["Yes", "No"], key="feedback")
                    if feedback:
                        st.write(f"Thanks for your feedback: {feedback}")
        
        
        
        st.markdown("---")
        st.subheader("Saved Prompt History")
        saved_prompts = get_all_prompts()
        if saved_prompts:
            for pid, content, created_at in saved_prompts:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"**#{pid}** ({created_at}): {content}")
                with col2:
                    if st.button("Reuse", key=f"reuse_db_{pid}"):
                        
                        st.session_state.custom_query_input = content
                        st.session_state.custom_query_version += 1
                        st.success("Prompt loaded into custom query input. Please select 'Custom Query' and click Submit Query.")
        else:
            st.info("No saved prompts yet.")
        
        if st.button("Clear History"):
            st.session_state.prompt_history = []
            clear_all_prompts()
            st.success("Prompt history cleared.")
