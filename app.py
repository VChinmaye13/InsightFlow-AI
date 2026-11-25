import io, os, re, json, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="InsightFlow AI", page_icon="🔎", layout="wide")

# Improve visibility of dropdowns and input fields
st.markdown("""
<style>

/* Dropdown container */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Selected dropdown text and placeholder */
div[data-baseweb="select"] span {
    color: #1e293b !important;
    opacity: 1 !important;
}

/* Dropdown input text */
div[data-baseweb="select"] input {
    color: #1e293b !important;
    caret-color: #1e293b !important;
}

/* Dropdown menu options */
ul[role="listbox"] li {
    color: #1e293b !important;
    background-color: #ffffff !important;
}

/* Hover effect for options */
ul[role="listbox"] li:hover {
    background-color: #eef2ff !important;
    color: #3730a3 !important;
}

/* General text input visibility */
div[data-baseweb="input"] input {
    color: #1e293b !important;
}

/* File uploader text */
[data-testid="stFileUploader"] {
    color: #1e293b !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- Theme ----------
st.markdown("""
<style>
:root { --ink:#162033; --muted:#6b7280; --accent:#5b6cff; }
.block-container { padding-top: 1.5rem; max-width: 1450px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10182b,#17243d); }
[data-testid="stSidebar"] * { color: #f4f6ff !important; }
.hero { padding: 1.45rem 1.6rem; border-radius: 20px; background: linear-gradient(120deg,#18243c,#384a91 70%,#6c62d8); color:white; margin-bottom:1rem; }
.hero h1 { color:white; margin:0; font-size:2.15rem; letter-spacing:-.04em; }
.hero p { color:#e4e8ff; margin:.45rem 0 0 0; font-size:1rem; }
.kpi { background:linear-gradient(145deg,#ffffff,#f6f7ff); border:1px solid #e7e9f4; border-radius:16px; padding:1rem 1.15rem; }
.kpi-label { color:#697188; font-size:.82rem; }
.kpi-value { color:#18233a; font-size:1.65rem; font-weight:750; }
.small-muted { color:#788196; font-size:.85rem; }
div[data-testid="stMetric"] { background:#f7f8fc; border:1px solid #e8eaf3; padding:14px 16px; border-radius:14px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:10px 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔎 InsightFlow AI</h1>
  <p>Your intelligent data analyst — explore datasets, ask questions, uncover patterns, and create reports.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def read_dataset(file_bytes, filename):
    name = filename.lower()
    if name.endswith(".csv"):
        try: return pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError: return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Please upload a CSV or Excel file.")

def clean_numeric(df):
    return df.select_dtypes(include=np.number)

def infer_date_columns(df):
    candidates=[]
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
        elif df[col].dtype == "object":
            sample=df[col].dropna().astype(str).head(100)
            if len(sample) and any(re.search(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?", x) for x in sample):
                parsed=pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > .65:
                    candidates.append(col)
    return candidates

def profile_dataframe(df):
    missing=df.isna().sum()
    return {
        "rows":len(df), "cols":len(df.columns),
        "missing_cells":int(missing.sum()),
        "missing_pct":float(missing.sum()/(max(1,len(df)*len(df.columns)))*100),
        "duplicates":int(df.duplicated().sum()),
        "numeric_cols":list(df.select_dtypes(include=np.number).columns),
        "categorical_cols":list(df.select_dtypes(include=["object","category","bool"]).columns),
        "date_cols":infer_date_columns(df)
    }

def suggest_questions(df):
    """Suggest analytical questions based on the uploaded dataset's columns."""
    questions = ["How many rows are there?", "How many columns are there?",
                 "Are there any missing values?", "How many duplicate rows are there?"]
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    dates = infer_date_columns(df)
    for col in numeric[:2]:
        questions.extend([f"What is the average {col}?", f"What is the maximum {col}?",
                         f"What is the minimum {col}?", f"What is the total {col}?"])
    if numeric and categorical:
        for col in categorical[:2]:
            questions.append(f"Which {col} has the highest {numeric[0]}?")
    if numeric and dates:
        questions.append(f"How has {numeric[0]} changed over time?")
    return list(dict.fromkeys(questions))[:14]


def simple_answer(question, df):
    """Transparent local Q&A for common aggregations; never executes user-generated code."""
    q = re.sub(r"[^a-z0-9_ ]+", " ", str(question).lower())
    q = re.sub(r"\s+", " ", q).strip()
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    columns = list(df.columns)

    def mentioned_column(c):
        name = re.sub(r"[^a-z0-9_ ]+", " ", str(c).lower()).strip()
        return bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q) is not None

    # Simple dataset-wide questions
    if any(x in q for x in ["how many rows", "row count", "number of rows", "record count"]):
        return f"The dataset contains **{len(df):,} rows**."
    if any(x in q for x in ["how many columns", "column count", "number of columns", "feature count"]):
        return f"The dataset contains **{len(columns)} columns**."
    if any(x in q for x in ["missing", "null", "empty cells"]):
        matched = next((c for c in columns if mentioned_column(c)), None)
        if matched is not None:
            n = int(df[matched].isna().sum())
            return f"Column **{matched}** has **{n:,} missing values** ({n/max(1,len(df))*100:.2f}%)."
        n = int(df.isna().sum().sum())
        return f"The dataset has **{n:,} missing cells** across all columns."
    if "duplicate" in q:
        return f"There are **{int(df.duplicated().sum()):,} duplicate rows**."

    # Numeric metric mentioned in the question
    metric = next((c for c in numeric_cols if mentioned_column(c)), None)
    category = next((c for c in columns if c not in numeric_cols and mentioned_column(c)), None)

    if metric is not None:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not len(series):
            return f"Column **{metric}** has no valid numeric values."

        if any(x in q for x in ["average", "mean", "avg"]):
            return f"Average **{metric}**: **{series.mean():,.3f}**."
        if "median" in q:
            return f"Median **{metric}**: **{series.median():,.3f}**."
        if any(x in q for x in ["minimum", "lowest", "min value", "smallest"]):
            return f"Minimum **{metric}**: **{series.min():,.3f}**."
        if any(x in q for x in ["maximum", "highest", "max value", "largest"]):
            # If a category is also named, aggregate the metric by that category.
            if category is not None and any(x in q for x in ["by", "which", "top", "highest", "most"]):
                grouped = df.groupby(category, dropna=False)[metric].sum().sort_values(ascending=False)
                if len(grouped):
                    best = grouped.index[0]
                    return f"**{best}** has the highest total **{metric}**: **{grouped.iloc[0]:,.3f}**."
            return f"Maximum **{metric}**: **{series.max():,.3f}**."
        if any(x in q for x in ["sum", "total"]):
            return f"Total **{metric}**: **{series.sum():,.3f}**."

    # Date-based trend and period ranking
    dates = infer_date_columns(df)
    date_col = next((c for c in dates if mentioned_column(c)), dates[0] if dates else None)
    if metric is not None and date_col is not None and any(x in q for x in ["over time", "trend", "period", "month", "year"]):
        work = df[[date_col, metric]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna()
        if not work.empty:
            work["_period"] = work[date_col].dt.to_period("M").astype(str)
            grouped = work.groupby("_period")[metric].sum().sort_index()
            if any(x in q for x in ["highest", "maximum", "peak", "top"]):
                peak = grouped.idxmax()
                return f"Highest **{metric}** occurred in **{peak}**: **{grouped.loc[peak]:,.3f}**."
            first_period, last_period = grouped.index[0], grouped.index[-1]
            first_value, last_value = float(grouped.iloc[0]), float(grouped.iloc[-1])
            change = last_value - first_value
            pct = (change / abs(first_value) * 100) if first_value else np.nan
            direction = "increased" if change > 0 else "decreased" if change < 0 else "remained stable"
            pct_text = f" ({pct:+.2f}%)" if pd.notna(pct) else ""
            return (f"**{metric}** {direction} from **{first_value:,.3f}** in {first_period} "
                    f"to **{last_value:,.3f}** in {last_period}; change **{change:+,.3f}**{pct_text}.")

    # Category-level ranking where a metric was not explicitly named
    if category is not None and any(x in q for x in ["top", "highest", "most", "which"]):
        counts = df[category].value_counts(dropna=False)
        return f"Most frequent **{category}** value: **{counts.index[0]}** ({int(counts.iloc[0]):,} rows)."

    return ("I can answer common questions locally: row/column count, missing values, duplicates, "
            "and numeric mean/median/min/max/sum. Include the exact column name in your question. "
            "For open-ended AI explanations, use the optional AI mode.")
def build_insights(df):
    insights=[]
    if len(df)==0: return ["Dataset is empty."]
    miss=df.isna().sum()
    if miss.sum():
        c=miss.idxmax()
        insights.append(f"**Missing data:** {int(miss.sum()):,} missing cells; `{c}` has the most ({int(miss[c]):,}).")
    else:
        insights.append("**Completeness:** no missing cells detected.")
    dup=int(df.duplicated().sum())
    insights.append(f"**Duplicates:** {dup:,} duplicate rows detected.")
    nums=df.select_dtypes(include=np.number)
    if len(nums.columns):
        v=nums.var(numeric_only=True).sort_values(ascending=False)
        if len(v) and pd.notna(v.iloc[0]):
            insights.append(f"**Variation:** `{v.index[0]}` has the highest numeric variance ({v.iloc[0]:,.3g}).")
        for c in nums.columns[:3]:
            s=nums[c].dropna()
            if len(s):
                insights.append(f"**{c}:** mean {s.mean():,.3g}; median {s.median():,.3g}; range {s.min():,.3g}–{s.max():,.3g}.")
    cats=df.select_dtypes(include=["object","category"])
    if len(cats.columns):
        c=cats.columns[0]
        top=df[c].value_counts(dropna=False).head(1)
        if len(top): insights.append(f"**Most common `{c}`:** `{top.index[0]}` ({int(top.iloc[0]):,} rows).")
    return insights

# Sidebar
with st.sidebar:
    st.markdown("## Workspace")
    st.caption("Choose a sample dataset or upload your own.")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    sample_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".xlsx", ".xls"))])
    sample_choice = st.selectbox("Explore a sample dataset", ["Select a sample…"] + sample_files)
    uploaded=st.file_uploader("Or upload CSV or Excel", type=["csv","xlsx","xls"])
    st.markdown("---")
    st.markdown("### Quick guide")
    st.markdown("1. Choose or upload a dataset\n2. Review data quality\n3. Explore charts\n4. Ask suggested questions\n5. Export a report")
    st.markdown("---")
    st.caption("InsightFlow AI • Data Analyst Copilot")

if uploaded:
    try:
        df=read_dataset(uploaded.getvalue(), uploaded.name)
        if df.empty: st.warning("This file contains no rows.")
        if len(df.columns)>100: st.warning("Large number of columns detected. Consider selecting a subset for easier exploration.")
        df.columns=[str(c) for c in df.columns]
        st.success(f"Loaded **{uploaded.name}** successfully.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
elif sample_choice != "Select a sample…":
    try:
        sample_path = os.path.join(data_dir, sample_choice)
        with open(sample_path, "rb") as sample_file:
            df=read_dataset(sample_file.read(), sample_choice)
        df.columns=[str(c) for c in df.columns]
        st.success(f"Loaded sample dataset: **{sample_choice}**")
    except Exception as e:
        st.error(f"Could not read sample dataset: {e}")
        st.stop()
else:
    # Demo dataset so the UI is immediately usable
    demo={
        "Date":pd.date_range("2024-01-01", periods=180, freq="D"),
        "Region":np.random.default_rng(12).choice(["North","South","East","West"],180),
        "Product":np.random.default_rng(21).choice(["Laptop","Phone","Tablet","Monitor"],180),
        "Units":np.random.default_rng(7).integers(1,35,180),
        "Revenue":np.round(np.random.default_rng(9).normal(850,220,180).clip(80),2),
        "Customer_Satisfaction":np.round(np.random.default_rng(15).uniform(2.5,5,180),1)
    }
    df=pd.DataFrame(demo)
    st.info("You're viewing a demo dataset. Select a sample dataset or upload your own from the sidebar.")

profile=profile_dataframe(df)
# overview KPI
k1,k2,k3,k4=st.columns(4)
k1.metric("Records",f"{profile['rows']:,}")
k2.metric("Features",f"{profile['cols']:,}")
k3.metric("Missing cells",f"{profile['missing_cells']:,}")
k4.metric("Duplicate rows",f"{profile['duplicates']:,}")

tabs=st.tabs(["📊 Overview","🧹 Data Quality","📈 Visual Explorer","💬 Ask InsightFlow","🧠 Anomaly Detection","📄 Report"])

with tabs[0]:
    st.subheader("Dataset overview")
    left,right=st.columns([1.4,1])
    with left:
        st.markdown("#### Data preview")
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Column types")
        type_df=pd.DataFrame({"Column":df.columns.astype(str),"Data type":df.dtypes.astype(str),"Non-null":df.notna().sum().values})
        st.dataframe(type_df,use_container_width=True,hide_index=True)
        st.markdown("#### Automated insights")
        for item in build_insights(df)[:5]: st.markdown("• "+item)
    nums=df.select_dtypes(include=np.number)
    if len(nums.columns)>=2:
        st.markdown("#### Correlation snapshot")
        corr=nums.corr(numeric_only=True)
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="Blues",title="Numeric feature correlation")
        st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.subheader("Data quality audit")
    missing=df.isna().sum().sort_values(ascending=False)
    q1,q2=st.columns(2)
    with q1:
        st.markdown("#### Missing values by column")
        missing_df=missing[missing>0].rename("Missing").reset_index().rename(columns={"index":"Column"})
        if len(missing_df):
            st.dataframe(missing_df,use_container_width=True,hide_index=True)
            st.plotly_chart(px.bar(missing_df,x="Column",y="Missing",title="Missing cells",color="Missing"),use_container_width=True)
        else: st.success("No missing values detected.")
    with q2:
        st.markdown("#### Duplicate records")
        st.metric("Duplicate rows",int(df.duplicated().sum()))
        st.caption("Duplicates are reported, not removed automatically, so the original data remains unchanged.")
        st.markdown("#### Numeric summary")
        if len(profile["numeric_cols"]):
            st.dataframe(df[profile["numeric_cols"]].describe().T,use_container_width=True)
        else: st.info("No numeric columns found.")

with tabs[2]:
    st.subheader("Build an interactive chart")
    numeric=list(df.select_dtypes(include=np.number).columns)
    categorical=list(df.select_dtypes(include=["object","category","bool"]).columns)
    chart_type=st.selectbox("Chart type",["Bar","Line","Scatter","Histogram","Box"])
    a,b,c=st.columns(3)
    x_col=a.selectbox("X / category",list(df.columns),index=0)
    y_col=b.selectbox("Y / numeric",numeric,index=0 if numeric else None) if numeric else None
    color_col=c.selectbox("Color by (optional)",["None"]+list(df.columns))
    color=None if color_col=="None" else color_col
    try:
        if chart_type=="Bar":
            if y_col: fig=px.bar(df,x=x_col,y=y_col,color=color,title=f"{y_col} by {x_col}")
            else: fig=px.bar(df[x_col].value_counts().head(20).reset_index(),x=x_col,y="count",title=f"Count by {x_col}")
        elif chart_type=="Line":
            if y_col: fig=px.line(df,x=x_col,y=y_col,color=color,title=f"{y_col} over {x_col}")
            else: st.warning("Select a numeric column for a line chart."); fig=None
        elif chart_type=="Scatter":
            if y_col and len(numeric)>1:
                x_num=st.selectbox("Scatter X numeric",numeric)
                fig=px.scatter(df,x=x_num,y=y_col,color=color,title=f"{y_col} vs {x_num}")
            else: st.warning("Scatter requires at least two numeric columns."); fig=None
        elif chart_type=="Histogram":
            hist_col=st.selectbox("Histogram column",numeric or list(df.columns))
            fig=px.histogram(df,x=hist_col,color=color,title=f"Distribution of {hist_col}")
        else:
            box_col=st.selectbox("Box plot column",numeric or list(df.columns))
            fig=px.box(df,y=box_col,x=x_col if x_col!=box_col else None,color=color,title=f"Distribution of {box_col}")
        if fig is not None: st.plotly_chart(fig,use_container_width=True)
    except Exception as e: st.warning(f"Chart could not be created with this combination: {e}")

with tabs[3]:
    st.subheader("Ask questions about your dataset")
    st.caption("Local mode answers common analytical questions without sending your data to an external service.")
    examples=suggest_questions(df)
    selected=st.selectbox("Suggested questions for this dataset",["Choose a question…"]+examples)
    question=st.text_input("Your question",value="" if selected=="Choose a question…" else selected,placeholder="Choose a suggested question or type your own")
    if st.button("Analyze question",type="primary"):
        if question.strip():
            answer=simple_answer(question,df)
            st.markdown("### Answer")
            st.markdown(answer)
        else: st.warning("Enter a question first.")
    st.markdown("---")
    st.markdown("#### AI mode (optional)")
    st.caption("Optional: use your own API key for AI explanations. Do not paste a shared/published key, and never commit keys to GitHub.")
    api_key=st.text_input("Your OpenAI API key (optional; session only)",type="password",key="llm_key")
    if api_key and question.strip() and st.button("Ask AI about this question"):
        try:
            from openai import OpenAI
            client=OpenAI(api_key=api_key)
            schema=df.head(5).to_csv(index=False)
            prompt=("You are a careful data analyst. Use only the dataset summary and sample below. "
                    "Do not claim to have computed values not present. If a calculation is needed, explain the query or ask the user to use the local analysis features.\n\n"
                    f"Columns and types: {dict(zip(df.columns,df.dtypes.astype(str)))}\nSample rows:\n{schema}\nQuestion: {question}")
            response=client.responses.create(model="gpt-4.1-mini",input=prompt)
            st.markdown(response.output_text)
        except Exception as e: st.error(f"AI request failed. Check the API key and network connection. Details: {e}")

with tabs[4]:
    st.subheader("Anomaly detection")
    st.caption(
        "Isolation Forest flags unusual records. The explanation below highlights "
        "which selected features differ most from the dataset's typical values; "
        "these are diagnostic clues, not proof of an error or causal explanation."
    )

    numeric = list(df.select_dtypes(include=np.number).columns)
    if len(numeric) < 1:
        st.info("Anomaly detection requires at least one numeric column.")
    else:
        selected_cols = st.multiselect(
            "Select numeric features",
            numeric,
            default=numeric[:min(5, len(numeric))],
            key="anomaly_features",
        )
        contamination = st.slider(
            "Expected anomaly proportion", 0.01, 0.20, 0.05, 0.01,
            key="anomaly_contamination",
        )

        if st.button("Detect anomalies", type="primary", key="run_anomaly_detection"):
            if not selected_cols:
                st.warning("Select at least one numeric feature.")
            else:
                work = df[selected_cols].replace([np.inf, -np.inf], np.nan)
                valid = work.dropna()

                if len(valid) < 10:
                    st.warning("At least 10 complete rows are recommended.")
                else:
                    model = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=100,
                    )
                    predictions = model.fit_predict(valid)
                    decision_values = model.decision_function(valid)

                    result = df.loc[valid.index].copy()
                    result["Anomaly"] = np.where(
                        predictions == -1, "Anomaly", "Normal"
                    )

                    # Convert the Isolation Forest decision values to a relative
                    # ranking score. It is NOT a probability of being fraudulent.
                    decision_rank = pd.Series(
                        decision_values, index=valid.index
                    ).rank(method="average", pct=True)
                    result["Anomaly Score (%)"] = (
                        (1.0 - decision_rank) * 100.0
                    ).round(1)

                    # Robust feature deviation: IQR-based scale is less sensitive
                    # to extreme values than standard deviation.
                    feature_stats = {}
                    for feature in selected_cols:
                        values = pd.to_numeric(valid[feature], errors="coerce")
                        q1 = float(values.quantile(0.25))
                        q3 = float(values.quantile(0.75))
                        median = float(values.median())
                        iqr = q3 - q1
                        robust_scale = iqr / 1.349 if iqr > 0 else float(values.std())
                        if not np.isfinite(robust_scale) or robust_scale <= 0:
                            robust_scale = 1.0
                        feature_stats[feature] = {
                            "median": median,
                            "q1": q1,
                            "q3": q3,
                            "lower_fence": q1 - 1.5 * iqr,
                            "upper_fence": q3 + 1.5 * iqr,
                            "scale": robust_scale,
                        }

                    def explain_anomaly(row):
                        deviations = {}
                        for feature in selected_cols:
                            value = float(row[feature])
                            stats = feature_stats[feature]
                            deviations[feature] = abs(value - stats["median"]) / stats["scale"]

                        ranked_features = sorted(
                            deviations, key=deviations.get, reverse=True
                        )[:3]
                        if not ranked_features:
                            return "No selected feature could be evaluated."

                        reason_parts = []
                        for feature in ranked_features:
                            value = float(row[feature])
                            stats = feature_stats[feature]
                            dev = deviations[feature]
                            if value < stats["lower_fence"] or value > stats["upper_fence"]:
                                reason_parts.append(
                                    f"{feature}={value:,.3g} is outside the IQR-based "
                                    f"range [{stats['lower_fence']:,.3g}, "
                                    f"{stats['upper_fence']:,.3g}]"
                                )
                            else:
                                reason_parts.append(
                                    f"{feature}={value:,.3g} differs from its median "
                                    f"({stats['median']:,.3g}) by {dev:.2f} robust scales"
                                )
                        return "; ".join(reason_parts)

                    anomaly_mask = result["Anomaly"] == "Anomaly"
                    result["Anomaly Reason"] = ""
                    result.loc[anomaly_mask, "Anomaly Reason"] = result.loc[
                        anomaly_mask, selected_cols
                    ].apply(explain_anomaly, axis=1)

                    flagged = result[anomaly_mask].sort_values(
                        "Anomaly Score (%)", ascending=False
                    )
                    st.metric("Flagged anomalies", int(anomaly_mask.sum()))
                    st.caption(
                        "Anomaly Score is a relative ranking within this dataset. "
                        "Higher scores indicate records ranked as more unusual; "
                        "the score is not a probability."
                    )

                    display_cols = [
                        "Anomaly Score (%)", "Anomaly Reason", *selected_cols, "Anomaly"
                    ]
                    st.markdown("#### Flagged records with explanations")
                    st.dataframe(
                        flagged[display_cols].head(100),
                        use_container_width=True,
                        hide_index=False,
                    )

                    if not flagged.empty:
                        st.markdown("#### Explain an individual anomaly")
                        chosen_index = st.selectbox(
                            "Select a flagged record",
                            options=flagged.index.tolist(),
                            format_func=lambda idx: (
                                f"Row {idx} · score "
                                f"{result.loc[idx, 'Anomaly Score (%)']:.1f}%"
                            ),
                            key="selected_anomaly_row",
                        )
                        chosen = result.loc[chosen_index]
                        score = float(chosen["Anomaly Score (%)"])
                        severity = "High" if score >= 95 else "Medium" if score >= 80 else "Low"
                        st.write(f"**Relative severity:** {severity}")
                        st.write(f"**Why it was flagged:** {chosen['Anomaly Reason']}")

                        detail_rows = []
                        for feature in selected_cols:
                            stats = feature_stats[feature]
                            value = float(chosen[feature])
                            detail_rows.append({
                                "Feature": feature,
                                "Observed value": value,
                                "Dataset median": stats["median"],
                                "IQR lower fence": stats["lower_fence"],
                                "IQR upper fence": stats["upper_fence"],
                                "Robust deviation": abs(value - stats["median"]) / stats["scale"],
                                "Outside IQR fence": (
                                    value < stats["lower_fence"] or value > stats["upper_fence"]
                                ),
                            })
                        details = pd.DataFrame(detail_rows).sort_values(
                            "Robust deviation", ascending=False
                        )
                        st.dataframe(details, use_container_width=True, hide_index=True)
                        st.caption(
                            "The feature ranking is based on robust distance from each "
                            "feature's median. Isolation Forest considers the combined "
                            "feature pattern, so this is an interpretable diagnostic "
                            "rather than a formal causal attribution."
                        )

                    st.download_button(
                        "Download flagged records with reasons",
                        flagged.to_csv(index=True).encode("utf-8"),
                        "insightflow_anomalies_explained.csv",
                        "text/csv",
                    )

                    if len(selected_cols) >= 2:
                        fig = px.scatter(
                            result,
                            x=selected_cols[0],
                            y=selected_cols[1],
                            color="Anomaly",
                            hover_data=["Anomaly Score (%)", "Anomaly Reason"],
                            title="Anomaly map (hover for explanation)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Export your analysis")
    st.write("Generate a concise report containing dataset details, data-quality checks, and automated insights.")
    report=[
        "# InsightFlow AI — Data Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source file: {uploaded.name if uploaded else 'Built-in sample dataset'}",
        "",
        "## Dataset summary",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns):,}",
        f"- Missing cells: {int(df.isna().sum().sum()):,}",
        f"- Duplicate rows: {int(df.duplicated().sum()):,}",
        "",
        "## Automated insights",
        *[f"- {x.replace('**','')}" for x in build_insights(df)],
        "",
        "## Numeric summary",
        df.select_dtypes(include=np.number).describe().to_string() if len(df.select_dtypes(include=np.number).columns) else "No numeric columns."
    ]
    report_text="\n".join(report)
    st.text_area("Report preview",report_text,height=300)
    st.download_button("Download report (.md)",report_text.encode("utf-8"),"InsightFlow_AI_Report.md","text/markdown",type="primary")
    st.download_button("Download analyzed dataset (.csv)",df.to_csv(index=False).encode("utf-8"),"insightflow_dataset.csv","text/csv")

st.markdown("---")
st.markdown('<p class="small-muted">InsightFlow AI · Intelligent Data Analyst Copilot · Exploratory analysis supports decisions but does not replace domain review.</p>',unsafe_allow_html=True)
