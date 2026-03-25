import os

import altair as alt
import pandas as pd
import streamlit as st

from utils import load_artifacts, load_training_metadata, predict_from_input

ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
ENHANCED_CSV = os.path.join(DATA_DIR, "enhanced_salary_data.csv")
REAL_CSV = os.path.join(DATA_DIR, "real_india_salary_dataset.csv")
PLOTS_DIR = os.path.join(ROOT, "plots")
IMAGES_DIR = os.path.join(ROOT, "images")
MODEL_COMPARISON_CSV = os.path.join(PLOTS_DIR, "model_comparison.csv")
PREDICTED_VS_ACTUAL_PNG = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
FEATURE_IMPORTANCE_PNG = os.path.join(PLOTS_DIR, "feature_importance.png")


@st.cache_data
def load_model_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_data
def load_comparison_dataset():
    # Prefer prepared dataset when available, otherwise fall back to the real dataset.
    if os.path.exists(ENHANCED_CSV):
        return pd.read_csv(ENHANCED_CSV), os.path.basename(ENHANCED_CSV)
    if os.path.exists(REAL_CSV):
        return pd.read_csv(REAL_CSV), os.path.basename(REAL_CSV)
    return pd.DataFrame(), "none"


@st.cache_data
def load_metadata():
    return load_training_metadata(MODEL_DIR)


@st.cache_data
def load_model_comparison():
    if os.path.exists(MODEL_COMPARISON_CSV):
        return pd.read_csv(MODEL_COMPARISON_CSV)
    return pd.DataFrame()


def format_inr(value):
    amount = int(round(float(value)))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) <= 3:
        return f"Rs. {sign}{digits}"

    last_three = digits[-3:]
    remaining = digits[:-3]
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)

    indian_number = ",".join(groups + [last_three]) if groups else last_three
    return f"Rs. {sign}{indian_number}"


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(230, 92, 0, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(0, 153, 102, 0.14), transparent 24%),
                linear-gradient(180deg, #0d1117 0%, #111827 100%);
        }
        .hero-card, .info-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(17, 24, 39, 0.78);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.22);
        }
        .hero-title {
            font-size: 2.25rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 0;
            color: #fff5eb;
        }
        .hero-subtitle {
            margin-top: 0.6rem;
            color: #d1d5db;
            font-size: 1rem;
        }
        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .pill {
            font-size: 0.88rem;
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            background: rgba(255,255,255,0.06);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #f59e0b;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .muted-note {
            color: #9ca3af;
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(metadata):
    source_label = metadata.get("source_file", "No training metadata")
    source_type = metadata.get("source_type", "unknown")
    row_count = metadata.get("rows", 0)
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="section-label">India Salary Intelligence</div>
            <h1 class="hero-title">Salary Prediction for the Indian Job Market</h1>
            <div class="hero-subtitle">
                Predict annual CTC using role, experience, location, company type, and industry.
                Drop a real CSV at <code>data/real_india_salary_dataset.csv</code> and rerun training to switch from the synthetic fallback.
            </div>
            <div class="pill-row">
                <div class="pill">Dataset source: {source_label}</div>
                <div class="pill">Mode: {source_type}</div>
                <div class="pill">Prepared rows: {row_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_options(model, encoder):
    master_options = {
        "education": [
            "Diploma",
            "B.Sc",
            "B.Com",
            "B.A",
            "B.Tech",
            "B.E",
            "BBA",
            "M.Sc",
            "M.Com",
            "M.A",
            "M.Tech",
            "MBA",
            "MCA",
            "PGDM",
            "PhD",
        ],
        "role": [
            "Software Engineer",
            "Data Scientist",
            "Data Analyst",
            "Product Manager",
            "Project Manager",
            "Manager",
            "Marketing Manager",
            "Sales",
            "HR",
            "Accounting",
            "Finance",
            "Research",
            "Designer",
            "Business Analyst",
            "DevOps Engineer",
            "QA Engineer",
            "Consultant",
            "Director",
            "Executive",
            "Others",
        ],
        "gender": ["Male", "Female", "Other"],
        "location": [
            "Bengaluru",
            "Mumbai",
            "Delhi NCR",
            "Hyderabad",
            "Pune",
            "Chennai",
            "Ahmedabad",
            "Chandigarh",
            "Indore",
            "Jaipur",
            "Kochi",
            "Coimbatore",
            "Bhubaneswar",
            "Lucknow",
            "Nagpur",
            "Mysuru",
            "Surat",
            "Vadodara",
        ],
        "company_type": ["Service", "Product", "Startup", "MNC", "Domestic"],
        "industry": [
            "IT Services",
            "Product Tech",
            "Fintech",
            "Analytics",
            "Banking",
            "Insurance",
            "Consulting",
            "Manufacturing",
            "Retail",
            "Consumer Goods",
            "Media",
            "EdTech",
            "Pharma",
            "Healthcare Tech",
            "SaaS",
            "E-commerce",
            "AI/ML",
            "Real Estate",
            "Gaming",
        ],
    }

    def merge_unique(base, extra):
        ordered = []
        seen = set()
        for value in base + extra:
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    try:
        if encoder is None and hasattr(model, "named_steps"):
            encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
        encoder_options = {
            "education": list(encoder.categories_[0]),
            "role": list(encoder.categories_[1]),
            "gender": list(encoder.categories_[2]),
            "location": list(encoder.categories_[3]),
            "company_type": list(encoder.categories_[4]),
            "industry": list(encoder.categories_[5]),
        }
        return {
            "education": merge_unique(master_options["education"], encoder_options["education"]),
            "role": merge_unique(master_options["role"], encoder_options["role"]),
            "gender": merge_unique(master_options["gender"], encoder_options["gender"]),
            "location": ["General"] + merge_unique(master_options["location"], encoder_options["location"]),
            "company_type": ["General"] + merge_unique(master_options["company_type"], encoder_options["company_type"]),
            "industry": ["General"] + merge_unique(master_options["industry"], encoder_options["industry"]),
        }
    except Exception:
        fallback = dict(master_options)
        fallback["location"] = ["General"] + fallback["location"]
        fallback["company_type"] = ["General"] + fallback["company_type"]
        fallback["industry"] = ["General"] + fallback["industry"]
        return fallback


def resolve_general_inputs(df_enh, job_role, location, company_type, industry):
    """Convert General selections into common values from dataset for stable predictions."""
    if df_enh.empty or "Job Title" not in df_enh.columns:
        resolved = {
            "location": "Bengaluru" if location == "General" else location,
            "company_type": "MNC" if company_type == "General" else company_type,
            "industry": "IT Services" if industry == "General" else industry,
        }
        return resolved

    role_df = df_enh[df_enh["Job Title"] == job_role]
    if role_df.empty:
        role_df = df_enh

    def mode_or_default(column, default):
        if column in role_df.columns and not role_df[column].dropna().empty:
            return str(role_df[column].mode(dropna=True).iloc[0])
        return default

    return {
        "location": mode_or_default("Location", "Bengaluru") if location == "General" else location,
        "company_type": mode_or_default("Company Type", "MNC") if company_type == "General" else company_type,
        "industry": mode_or_default("Industry", "IT Services") if industry == "General" else industry,
    }


def build_market_slice(df_enh, job_role, location, company_type, industry):
    market_slice = df_enh[df_enh["Job Title"] == job_role]
    if {"Location", "Company Type", "Industry"}.issubset(df_enh.columns):
        role_filter = df_enh["Job Title"] == job_role
        location_filter = True if location == "General" else (df_enh["Location"] == location)
        company_filter = True if company_type == "General" else (df_enh["Company Type"] == company_type)
        industry_filter = True if industry == "General" else (df_enh["Industry"] == industry)

        market_slice = df_enh[role_filter & location_filter & company_filter & industry_filter]
        if len(market_slice) < 10:
            market_slice = df_enh[
                (df_enh["Job Title"] == job_role)
                & (True if location == "General" else (df_enh["Location"] == location))
                & (True if industry == "General" else (df_enh["Industry"] == industry))
            ]
        if len(market_slice) < 10:
            market_slice = df_enh[
                (df_enh["Job Title"] == job_role)
                & (True if location == "General" else (df_enh["Location"] == location))
            ]
        if len(market_slice) < 10:
            market_slice = df_enh[df_enh["Job Title"] == job_role]
    return market_slice


def render_model_insights(metadata):
    st.write("")
    st.markdown('<div class="section-label">Model Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "These plots explain model selection and behavior using training outputs from `plots/` and project EDA graphs."
    )

    comp = load_model_comparison()
    if not comp.empty and {"name", "r2", "mae", "rmse"}.issubset(comp.columns):
        comp_display = comp.copy()
        comp_display["mae"] = comp_display["mae"].apply(format_inr)
        comp_display["rmse"] = comp_display["rmse"].apply(format_inr)
        comp_display["r2"] = comp_display["r2"].round(4)

        st.dataframe(
            comp_display.rename(
                columns={
                    "name": "Model",
                    "r2": "R2",
                    "mae": "MAE",
                    "rmse": "RMSE",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_cols = st.columns(2)
        rmse_df = comp.sort_values("rmse", ascending=True)
        r2_df = comp.sort_values("r2", ascending=False)
        with chart_cols[0]:
            st.markdown("**RMSE Comparison (Lower is Better)**")
            st.bar_chart(rmse_df.set_index("name")["rmse"])
        with chart_cols[1]:
            st.markdown("**R2 Comparison (Higher is Better)**")
            st.bar_chart(r2_df.set_index("name")["r2"])

    image_cols = st.columns(2)
    with image_cols[0]:
        if os.path.exists(PREDICTED_VS_ACTUAL_PNG):
            st.image(PREDICTED_VS_ACTUAL_PNG, caption="Predicted vs Actual Salary")
        else:
            st.info("`plots/predicted_vs_actual.png` not found.")
    with image_cols[1]:
        if os.path.exists(FEATURE_IMPORTANCE_PNG):
            st.image(FEATURE_IMPORTANCE_PNG, caption="Feature Importance (Selected Model)")
        else:
            st.info("`plots/feature_importance.png` not found.")

    if os.path.isdir(IMAGES_DIR):
        eda_images = sorted(
            [
                os.path.join(IMAGES_DIR, f)
                for f in os.listdir(IMAGES_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if eda_images:
            with st.expander("Show Project EDA Graphs"):
                for img_path in eda_images:
                    st.image(img_path, caption=os.path.basename(img_path))

    if metadata:
        st.caption(
            f"Selected model: `{metadata.get('best_model', 'unknown')}` | "
            f"Training source: `{metadata.get('source_file', 'unknown')}` | "
            f"Rows: `{metadata.get('rows', 'unknown')}`"
        )


def main():
    st.set_page_config(page_title="India Salary Predictor", page_icon="INR", layout="wide")
    inject_styles()

    df_enh, comparison_dataset_name = load_comparison_dataset()
    metadata = load_metadata()
    model, encoder, scaler, columns = load_model_artifacts()
    options = get_options(model, encoder)

    render_header(metadata)
    st.write("")

    sidebar = st.sidebar
    sidebar.header("Dataset Import")
    sidebar.write("Place a real CSV at `data/real_india_salary_dataset.csv` and rerun `python train.py`.")
    sidebar.caption(
        "Accepted columns are flexible. The importer can map common names like role/designation, city/location, CTC/salary, company type, and industry."
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="section-label">Candidate Profile</div>', unsafe_allow_html=True)
        form_col1, form_col2 = st.columns(2, gap="medium")
        with form_col1:
            age = st.slider("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", options=options["gender"])
            location = st.selectbox("Location", options=options["location"])
            education = st.selectbox("Education Level", options=options["education"])
        with form_col2:
            experience = st.number_input(
                "Years of Experience", min_value=0.0, max_value=50.0, value=3.0, step=0.5
            )
            job_role = st.selectbox("Job Role", options=options["role"])
            company_type = st.selectbox("Company Type", options=options["company_type"])
            industry = st.selectbox("Industry", options=options["industry"])

        predict_clicked = st.button("Predict Salary", type="primary", use_container_width=True)
        st.caption("Tip: dropdowns include a broader India-focused catalog plus values seen during training.")

    with right:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">How To Improve Accuracy</div>
                <p>Use a real India salary CSV when possible. The app trains best when the dataset includes:</p>
                <ul>
                    <li>job title or designation</li>
                    <li>annual salary or CTC</li>
                    <li>years of experience</li>
                    <li>city or location</li>
                    <li>company type or employer category</li>
                    <li>industry or sector</li>
                </ul>
                <div class="muted-note">
                    If a real CSV is missing some fields, the importer fills gaps with India-specific defaults so training can still run.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_clicked:
        resolved = resolve_general_inputs(df_enh, job_role, location, company_type, industry)

        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Education Level": education,
                    "Job Title": job_role,
                    "Years of Experience": experience,
                    "Location": resolved["location"],
                    "Company Type": resolved["company_type"],
                    "Industry": resolved["industry"],
                }
            ]
        )

        preds = predict_from_input(model, encoder, scaler, columns, input_df)
        annual_salary = float(preds[0])
        monthly_salary = annual_salary / 12.0
        model_used = metadata.get("best_model", type(model).__name__)

        market_slice = pd.DataFrame()
        market_average = None
        if not df_enh.empty and "Job Title" in df_enh.columns and job_role in df_enh["Job Title"].unique():
            market_slice = build_market_slice(df_enh, job_role, location, company_type, industry)
            if not market_slice.empty:
                market_average = float(market_slice["Salary"].mean())

        st.write("")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Predicted Annual Salary", format_inr(annual_salary))
        metric_col2.metric("Estimated Monthly Salary", format_inr(monthly_salary))
        metric_col3.metric(
            "Market Sample Size",
            f"{len(market_slice)}",
            help="Number of similar records used to compute the comparison average.",
        )
        metric_col4.metric("Model Used", model_used)

        bottom_left, bottom_right = st.columns([0.92, 1.08], gap="large")
        with bottom_left:
            st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
            summary_df = pd.DataFrame(
                {
                    "Field": [
                        "Age",
                        "Gender",
                        "Education Level",
                        "Job Title",
                        "Years of Experience",
                        "Location",
                        "Company Type",
                        "Industry",
                    ],
                    "Value": [
                        age,
                        gender,
                        education,
                        job_role,
                        f"{experience:.1f}",
                        location,
                        company_type,
                        industry,
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            if "General" in {location, company_type, industry}:
                st.caption(
                    f"`General` resolved for prediction as Location `{resolved['location']}`, Company Type `{resolved['company_type']}`, Industry `{resolved['industry']}`."
                )

        with bottom_right:
            st.markdown('<div class="section-label">Market Comparison</div>', unsafe_allow_html=True)
            if market_average is None:
                st.info("No comparison dataset is available yet. Train the model first to generate `enhanced_salary_data.csv`.")
            else:
                comp = pd.DataFrame(
                    {
                        "Label": [f"Market average for {job_role}", "Predicted salary"],
                        "Salary (INR)": [market_average, annual_salary],
                    }
                )
                comp_display = comp.copy()
                comp_display["Salary (INR)"] = comp_display["Salary (INR)"].apply(format_inr)
                st.dataframe(
                    comp_display,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"Comparison based on {len(market_slice)} similar record(s). Filtering prefers role + location + company type + industry, then falls back gracefully."
                )
                chart = (
                    alt.Chart(comp)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    .encode(
                        x=alt.X(
                            "Label:N",
                            sort=None,
                            axis=alt.Axis(labelAngle=0, labelLimit=240, labelOverlap=False),
                            title=None,
                        ),
                        y=alt.Y("Salary (INR):Q", title="Salary (INR)"),
                        color=alt.Color(
                            "Label:N",
                            scale=alt.Scale(
                                domain=[f"Market average for {job_role}", "Predicted salary"],
                                range=["#f59e0b", "#38bdf8"],
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Label:N", title="Type"),
                            alt.Tooltip("Salary (INR):Q", format=",.0f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

    render_model_insights(metadata)

    st.caption(
        f"Model artifacts loaded from `model/`. Comparison dataset: `{comparison_dataset_name}`. Training source: `{metadata.get('source_file', 'unknown')}`."
    )


if __name__ == "__main__":
    main()
