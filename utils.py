import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAT_COLS = ["Education Level", "Job Title", "Gender", "Location", "Company Type", "Industry"]
NUM_COLS = ["Age", "Years of Experience"]
STANDARD_COLS = NUM_COLS + CAT_COLS + ["Salary"]

ROLE_MAP = {
    "software": "Software Engineer",
    "developer": "Software Engineer",
    "engineer": "Software Engineer",
    "data scientist": "Data Scientist",
    "machine learning": "Data Scientist",
    "data": "Data Analyst",
    "analyst": "Data Analyst",
    "product": "Product Manager",
    "marketing": "Marketing Manager",
    "sales": "Sales",
    "hr": "HR",
    "human resources": "HR",
    "account": "Accounting",
    "finance": "Finance",
    "research": "Research",
    "designer": "Designer",
    "ux": "Designer",
    "director": "Director",
    "ceo": "Executive",
    "cto": "Executive",
    "vp": "Executive",
    "manager": "Manager",
}

INDUSTRY_OPTIONS = {
    "Software Engineer": ["IT Services", "Product Tech", "Fintech", "E-commerce", "SaaS"],
    "Data Scientist": ["Analytics", "AI/ML", "Fintech", "E-commerce", "Healthcare Tech"],
    "Data Analyst": ["Analytics", "IT Services", "Banking", "Retail", "Consulting"],
    "Product Manager": ["Product Tech", "Fintech", "E-commerce", "SaaS"],
    "Manager": ["IT Services", "Banking", "Manufacturing", "Consulting", "Retail"],
    "Marketing Manager": ["Consumer Goods", "Media", "E-commerce", "EdTech"],
    "Sales": ["Retail", "Consumer Goods", "SaaS", "Real Estate"],
    "HR": ["IT Services", "Consulting", "Manufacturing", "Healthcare"],
    "Accounting": ["Banking", "Manufacturing", "Retail", "Consulting"],
    "Finance": ["Banking", "Fintech", "Consulting", "Insurance"],
    "Research": ["Pharma", "AI/ML", "Healthcare Tech", "EdTech"],
    "Designer": ["Product Tech", "Media", "E-commerce", "Gaming"],
    "Director": ["IT Services", "Banking", "Consulting", "Manufacturing"],
    "Executive": ["Banking", "Manufacturing", "Consulting", "Conglomerate"],
    "Others": ["Services", "Retail", "Manufacturing"],
}

LOCATION_MODIFIER = {
    "Bengaluru": 1.18,
    "Mumbai": 1.16,
    "Delhi NCR": 1.15,
    "Hyderabad": 1.12,
    "Pune": 1.10,
    "Chennai": 1.07,
    "Ahmedabad": 0.95,
    "Chandigarh": 0.97,
    "Indore": 0.92,
    "Jaipur": 0.90,
    "Kochi": 0.93,
    "Coimbatore": 0.92,
    "Bhubaneswar": 0.88,
    "Lucknow": 0.88,
    "Nagpur": 0.87,
    "Mysuru": 0.89,
    "Surat": 0.89,
    "Vadodara": 0.88,
}

COMPANY_TYPE_MODIFIER = {
    "Startup": 1.06,
    "Product": 1.15,
    "Service": 0.94,
    "MNC": 1.12,
    "Domestic": 0.96,
}

INDUSTRY_MODIFIER = {
    "Product Tech": 1.16,
    "SaaS": 1.14,
    "Fintech": 1.14,
    "AI/ML": 1.13,
    "Healthcare Tech": 1.08,
    "Analytics": 1.07,
    "E-commerce": 1.08,
    "IT Services": 0.98,
    "Consulting": 1.05,
    "Banking": 1.08,
    "Insurance": 1.0,
    "Manufacturing": 0.95,
    "Retail": 0.92,
    "Consumer Goods": 0.94,
    "Media": 0.93,
    "EdTech": 0.97,
    "Pharma": 1.02,
    "Gaming": 1.03,
    "Real Estate": 0.95,
    "Conglomerate": 1.04,
    "Services": 0.92,
}

ROLE_SALARY_BANDS = {
    "Software Engineer": {"entry": (450000, 900000), "mid": (800000, 1800000), "senior": (1600000, 3800000)},
    "Data Scientist": {"entry": (600000, 1200000), "mid": (1000000, 2200000), "senior": (2000000, 4200000)},
    "Data Analyst": {"entry": (350000, 700000), "mid": (600000, 1200000), "senior": (1200000, 2400000)},
    "Product Manager": {"entry": (700000, 1400000), "mid": (1400000, 2600000), "senior": (2600000, 5000000)},
    "Manager": {"entry": (550000, 1100000), "mid": (1100000, 2200000), "senior": (2200000, 4200000)},
    "Marketing Manager": {"entry": (400000, 800000), "mid": (700000, 1400000), "senior": (1400000, 2600000)},
    "Sales": {"entry": (300000, 700000), "mid": (500000, 1200000), "senior": (1000000, 2200000)},
    "HR": {"entry": (300000, 650000), "mid": (500000, 1100000), "senior": (1000000, 2200000)},
    "Accounting": {"entry": (280000, 550000), "mid": (450000, 900000), "senior": (900000, 1800000)},
    "Finance": {"entry": (400000, 800000), "mid": (700000, 1500000), "senior": (1400000, 3000000)},
    "Research": {"entry": (400000, 850000), "mid": (700000, 1500000), "senior": (1400000, 2800000)},
    "Designer": {"entry": (350000, 700000), "mid": (600000, 1300000), "senior": (1200000, 2400000)},
    "Director": {"entry": (1800000, 3000000), "mid": (2800000, 5000000), "senior": (4000000, 8000000)},
    "Executive": {"entry": (2500000, 4500000), "mid": (4000000, 8000000), "senior": (6000000, 10000000)},
    "Others": {"entry": (250000, 500000), "mid": (400000, 850000), "senior": (800000, 1600000)},
}

EDU_MODIFIER = {
    "Diploma": 0.88,
    "B.Tech": 1.0,
    "B.Sc": 0.92,
    "MCA": 1.04,
    "M.Tech": 1.10,
    "MBA": 1.12,
    "PhD": 1.15,
}


def normalize_text(value):
    return str(value).strip() if pd.notna(value) else ""


def normalize_gender(value):
    v = normalize_text(value).lower()
    if v in {"male", "m", "man"}:
        return "Male"
    if v in {"female", "f", "woman"}:
        return "Female"
    return "Other"


def normalize_education(value):
    v = normalize_text(value).lower()
    if not v:
        return random.choice(["B.Tech", "B.Sc"])
    if "ph" in v or "doctor" in v:
        return "PhD"
    if "mba" in v:
        return "MBA"
    if "mca" in v:
        return "MCA"
    if "m.tech" in v or "mtech" in v or "master" in v:
        return "M.Tech"
    if "diploma" in v:
        return "Diploma"
    if "b.sc" in v or "bsc" in v or "science" in v:
        return "B.Sc"
    if "b.tech" in v or "btech" in v or "b.e" in v or "be " in f"{v} " or "bachelor" in v:
        return "B.Tech"
    return random.choice(["B.Tech", "B.Sc", "MBA"])


def normalize_role(title):
    if pd.isna(title):
        return "Others"
    t = str(title).lower()
    for key, value in ROLE_MAP.items():
        if key in t:
            return value
    return "Others"


def normalize_location(value):
    v = normalize_text(value).lower()
    city_aliases = {
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "bombay": "Mumbai",
        "mumbai": "Mumbai",
        "delhi": "Delhi NCR",
        "gurgaon": "Delhi NCR",
        "gurugram": "Delhi NCR",
        "noida": "Delhi NCR",
        "delhi ncr": "Delhi NCR",
        "hyderabad": "Hyderabad",
        "pune": "Pune",
        "chennai": "Chennai",
        "ahmedabad": "Ahmedabad",
        "chandigarh": "Chandigarh",
        "indore": "Indore",
        "jaipur": "Jaipur",
        "kochi": "Kochi",
        "coimbatore": "Coimbatore",
        "bhubaneswar": "Bhubaneswar",
        "lucknow": "Lucknow",
        "nagpur": "Nagpur",
        "mysore": "Mysuru",
        "mysuru": "Mysuru",
        "surat": "Surat",
        "vadodara": "Vadodara",
    }
    for alias, canonical in city_aliases.items():
        if alias in v:
            return canonical
    return ""


def normalize_company_type(value):
    v = normalize_text(value).lower()
    if "mnc" in v or "multinational" in v:
        return "MNC"
    if "startup" in v or "start-up" in v:
        return "Startup"
    if "product" in v:
        return "Product"
    if "service" in v or "services" in v or "outsourcing" in v:
        return "Service"
    if "domestic" in v or "indian" in v or "local" in v:
        return "Domestic"
    return ""


def normalize_industry(value):
    v = normalize_text(value).lower()
    alias_map = {
        "product": "Product Tech",
        "saas": "SaaS",
        "fintech": "Fintech",
        "ai": "AI/ML",
        "ml": "AI/ML",
        "analytics": "Analytics",
        "e-commerce": "E-commerce",
        "ecommerce": "E-commerce",
        "it service": "IT Services",
        "consult": "Consulting",
        "bank": "Banking",
        "insurance": "Insurance",
        "manufact": "Manufacturing",
        "retail": "Retail",
        "consumer": "Consumer Goods",
        "media": "Media",
        "edtech": "EdTech",
        "pharma": "Pharma",
        "game": "Gaming",
        "real estate": "Real Estate",
        "health": "Healthcare Tech",
        "conglomerate": "Conglomerate",
        "service": "Services",
    }
    for alias, canonical in alias_map.items():
        if alias in v:
            return canonical
    return ""


def choose_default_location(role):
    tier1 = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai"]
    tier2 = ["Ahmedabad", "Chandigarh", "Indore", "Jaipur", "Kochi", "Coimbatore"]
    tier3 = ["Bhubaneswar", "Lucknow", "Nagpur", "Mysuru", "Surat", "Vadodara"]
    options = tier1 + tier2 + tier3
    if role == "Software Engineer":
        weights = [0.13, 0.08, 0.13, 0.11, 0.10, 0.08, 0.08, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.00]
    elif role == "Data Scientist":
        weights = [0.16, 0.09, 0.14, 0.12, 0.10, 0.07, 0.07, 0.04, 0.03, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02, 0.02, 0.01, 0.00]
    else:
        weights = [0.12, 0.09, 0.12, 0.10, 0.09, 0.07, 0.07, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01]
    return random.choices(options, weights=weights, k=1)[0]


def choose_default_company_type(role, exp):
    exp = float(exp) if not pd.isna(exp) else 0.0
    if role in {"Software Engineer", "Data Scientist", "Product Manager"}:
        options = ["Service", "Product", "Startup", "MNC"]
        weights = [0.28, 0.28, 0.18, 0.26] if exp < 6 else [0.24, 0.28, 0.12, 0.36]
    elif role in {"Finance", "Accounting", "HR", "Sales", "Marketing Manager"}:
        options = ["Domestic", "MNC", "Startup"]
        weights = [0.5, 0.35, 0.15]
    else:
        options = ["Domestic", "MNC", "Startup"]
        weights = [0.45, 0.35, 0.20]
    return random.choices(options, weights=weights, k=1)[0]


def choose_default_industry(role):
    return random.choice(INDUSTRY_OPTIONS.get(role, INDUSTRY_OPTIONS["Others"]))


def estimate_salary_from_profile(row, add_noise=False):
    """Estimate INR salary from profile with balanced multi-factor influence."""
    exp = float(row.get("Years of Experience", 0.0)) if pd.notna(row.get("Years of Experience", 0.0)) else 0.0
    edu = row.get("Education Level", "B.Tech")
    role = row.get("Job Title", "Others")
    location = row.get("Location", "Bengaluru")
    company_type = row.get("Company Type", "MNC")
    industry = row.get("Industry", "IT Services")

    if exp <= 2:
        level = "entry"
        progress = min(exp / 2.0, 1.0)
    elif exp <= 10:
        level = "mid"
        progress = min(max((exp - 2.0) / 8.0, 0.0), 1.0)
    else:
        level = "senior"
        progress = min(max((exp - 10.0) / 20.0, 0.0), 1.0)

    low, high = ROLE_SALARY_BANDS.get(role, ROLE_SALARY_BANDS["Others"])[level]
    role_mid = (low + high) / 2.0
    # Keep experience effect moderate so other profile factors contribute meaningfully.
    level_shift = {"entry": 0.84, "mid": 1.0, "senior": 1.18}
    exp_curve = 0.86 + (0.28 * progress)
    base = role_mid * level_shift[level] * exp_curve

    salary = (
        base
        * (EDU_MODIFIER.get(edu, 1.0) ** 1.05)
        * (LOCATION_MODIFIER.get(location, 1.0) ** 1.20)
        * (COMPANY_TYPE_MODIFIER.get(company_type, 1.0) ** 1.22)
        * (INDUSTRY_MODIFIER.get(industry, 1.0) ** 1.22)
    )
    if role in {"Software Engineer", "Data Scientist", "Product Manager", "Finance"}:
        salary *= 1.04

    if add_noise:
        salary += np.random.normal(loc=0.0, scale=0.04 * salary)

    return int(min(max(180000, salary), 10000000))


def parse_salary_to_inr(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        num = float(value)
        if num <= 0:
            return np.nan
        if num <= 100:
            return num * 100000
        if num <= 1000:
            return num * 1000
        return num

    text = str(value).strip().lower().replace(",", "")
    if not text:
        return np.nan
    multiplier = 1
    if "crore" in text or "cr" in text:
        multiplier = 10000000
    elif "lakh" in text or "lac" in text or "lpa" in text:
        multiplier = 100000
    elif "k" in text:
        multiplier = 1000

    cleaned = []
    dot_seen = False
    for ch in text:
        if ch.isdigit():
            cleaned.append(ch)
        elif ch == "." and not dot_seen:
            cleaned.append(ch)
            dot_seen = True
    if not cleaned:
        return np.nan
    number = float("".join(cleaned))
    if multiplier != 1:
        return number * multiplier
    if number <= 100:
        return number * 100000
    if number <= 1000:
        return number * 1000
    return number


def find_matching_column(df, aliases):
    normalized_cols = {col.lower().strip(): col for col in df.columns}
    for alias in aliases:
        if alias in normalized_cols:
            return normalized_cols[alias]
    for alias in aliases:
        for key, original in normalized_cols.items():
            if alias in key:
                return original
    return None


def infer_age_from_experience(exp_series):
    exp = pd.to_numeric(exp_series, errors="coerce").fillna(0.0)
    age = (exp + np.random.uniform(20, 24, size=len(exp))).round(0)
    return age.clip(18, 60)


def build_real_dataset(raw_csv, output_csv):
    df = pd.read_csv(raw_csv)
    col_aliases = {
        "Age": ["age", "current age"],
        "Gender": ["gender", "sex"],
        "Education Level": ["education level", "education", "qualification", "degree"],
        "Job Title": ["job title", "designation", "role", "title", "job role"],
        "Years of Experience": ["years of experience", "experience", "total experience", "work experience", "exp"],
        "Salary": ["salary", "annual salary", "ctc", "annual ctc", "fixed ctc", "compensation"],
        "Location": ["location", "city", "job location"],
        "Company Type": ["company type", "company category", "employer type"],
        "Industry": ["industry", "sector", "domain"],
    }

    out = pd.DataFrame()
    for standard_col, aliases in col_aliases.items():
        match = find_matching_column(df, aliases)
        out[standard_col] = df[match] if match else np.nan

    out["Years of Experience"] = pd.to_numeric(out["Years of Experience"], errors="coerce")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Age"] = out["Age"].fillna(infer_age_from_experience(out["Years of Experience"]))
    out["Age"] = out["Age"].clip(18, 70)
    out["Years of Experience"] = out["Years of Experience"].fillna(out["Years of Experience"].median())
    out["Years of Experience"] = out["Years of Experience"].fillna(0.0).clip(0, 45)

    out["Gender"] = out["Gender"].apply(normalize_gender)
    out["Education Level"] = out["Education Level"].apply(normalize_education)
    out["Job Title"] = out["Job Title"].apply(normalize_role)
    out["Location"] = out["Location"].apply(normalize_location)
    out["Company Type"] = out["Company Type"].apply(normalize_company_type)
    out["Industry"] = out["Industry"].apply(normalize_industry)
    out["Salary"] = out["Salary"].apply(parse_salary_to_inr)

    out["Location"] = out.apply(
        lambda row: row["Location"] or choose_default_location(row["Job Title"]), axis=1
    )
    out["Company Type"] = out.apply(
        lambda row: row["Company Type"] or choose_default_company_type(row["Job Title"], row["Years of Experience"]),
        axis=1,
    )
    out["Industry"] = out.apply(
        lambda row: row["Industry"] or choose_default_industry(row["Job Title"]), axis=1
    )

    out = out.dropna(subset=["Salary", "Job Title"])
    # Blend reported salary with a calibrated India-profile estimate so model signal
    # isn't overwhelmingly dominated by experience-only patterns from source data.
    out["Calibrated Salary"] = out.apply(
        lambda row: estimate_salary_from_profile(row, add_noise=False), axis=1
    )
    out["Salary"] = 0.20 * out["Salary"].astype(float) + 0.80 * out["Calibrated Salary"].astype(float)
    out = out.drop(columns=["Calibrated Salary"])
    out["Salary"] = out["Salary"].clip(180000, 10000000).round(0).astype(int)
    out["Years of Experience"] = out["Years of Experience"].round(1)
    out = out[(out["Age"] >= 18) & (out["Age"] <= 70)]
    out = out[STANDARD_COLS]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def enhance_dataset(input_csv, output_csv, random_state: int = 42):
    """Build a synthetic India-shaped dataset from a generic salary CSV."""
    random.seed(random_state)
    np.random.seed(random_state)

    df = pd.read_csv(input_csv)
    df = df.dropna(how="all")

    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = np.nan

    if df["Years of Experience"].dtype == object:
        df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
    df["Years of Experience"] = df["Years of Experience"].fillna(df["Years of Experience"].median())
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())

    df["Education Level"] = df["Education Level"].apply(normalize_education)
    df["Gender"] = df["Gender"].apply(normalize_gender)
    df["Job Title"] = df["Job Title"].apply(normalize_role)
    df["Location"] = df.apply(
        lambda row: normalize_location(row["Location"]) or choose_default_location(row["Job Title"]), axis=1
    )
    df["Company Type"] = df.apply(
        lambda row: normalize_company_type(row["Company Type"])
        or choose_default_company_type(row["Job Title"], row["Years of Experience"]),
        axis=1,
    )
    df["Industry"] = df.apply(
        lambda row: normalize_industry(row["Industry"]) or choose_default_industry(row["Job Title"]), axis=1
    )

    df["Salary"] = df.apply(lambda row: estimate_salary_from_profile(row, add_noise=True), axis=1)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 70)]
    df["Years of Experience"] = df["Years of Experience"].round(1)
    df = df[STANDARD_COLS]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def prepare_training_dataset(input_csv, output_csv, source_type="synthetic", random_state=42):
    if source_type == "real":
        return build_real_dataset(input_csv, output_csv)
    return enhance_dataset(input_csv, output_csv, random_state=random_state)


def preprocess(df, encoder: OneHotEncoder = None, scaler: StandardScaler = None, fit: bool = False):
    cat_cols = CAT_COLS
    num_cols = NUM_COLS

    df_cat = df[cat_cols].fillna("Unknown")
    df_num = df[num_cols].fillna(0.0)

    if fit or encoder is None:
        try:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_arr = encoder.fit_transform(df_cat)
    else:
        enc_arr = encoder.transform(df_cat)

    if fit or scaler is None:
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(df_num)
    else:
        num_arr = scaler.transform(df_num)

    cat_columns = list(encoder.get_feature_names_out(cat_cols))
    feature_columns = num_cols + cat_columns
    X = np.hstack([num_arr, enc_arr])
    return X, encoder, scaler, feature_columns


def build_preprocessor():
    try:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", encoder, CAT_COLS),
        ]
    )


def get_feature_columns_from_preprocessor(preprocessor):
    cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
    return NUM_COLS + cat_features


def predict_from_input(model, encoder, scaler, columns, input_df):
    if hasattr(model, "named_steps"):
        return model.predict(input_df)

    X, _, _, feature_columns = preprocess(input_df, encoder=encoder, scaler=scaler, fit=False)
    X_df = pd.DataFrame(X, columns=feature_columns)
    for c in columns:
        if c not in X_df.columns:
            X_df[c] = 0.0
    X_df = X_df[columns]
    return model.predict(X_df.values)


def save_artifacts(model, scaler=None, columns=None, folder="model", encoder=None):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, "best_model.pkl"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    if columns is not None:
        joblib.dump(columns, os.path.join(folder, "columns.pkl"))
    if encoder is not None:
        joblib.dump(encoder, os.path.join(folder, "encoder.pkl"))


def save_training_metadata(folder, metadata):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "training_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def load_training_metadata(folder="model"):
    path = os.path.join(folder, "training_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_artifacts(folder="model"):
    model = joblib.load(os.path.join(folder, "best_model.pkl"))
    scaler_path = os.path.join(folder, "scaler.pkl")
    columns_path = os.path.join(folder, "columns.pkl")
    encoder_path = os.path.join(folder, "encoder.pkl")

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    columns = joblib.load(columns_path) if os.path.exists(columns_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    return model, encoder, scaler, columns
