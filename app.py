# app.py — Census Data Dashboard (Streamlit) — no separate search bar
# Deploy: push app.py + requirements.txt to GitHub and deploy on Streamlit Cloud
# or upload both files directly. No user API key needed.

from typing import Optional, Dict, List
from urllib.parse import urlencode

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
#st.set_page_config(page_title="Census Data Dashboard by Shah", layout="wide")

st.title("Census Data Dashboard — by Shah")

# Embedded API key so end users don't need one
API_KEY = "6c14346c155c7ae9110a833a854582dc60c3afd0"

# ---------------------------
# Dataset catalog
# ---------------------------
DATASETS: List[Dict] = [
    {"label": "ACS 1-year", "slug": "acs/acs1", "years": list(range(2005, 2024)), "geos": ["nation", "state", "county", "tract"]},
    {"label": "ACS 5-year", "slug": "acs/acs5", "years": list(range(2009, 2024)), "geos": ["nation", "state", "county", "tract"]},
    {"label": "Decennial 1990 SF1", "slug": "dec/sf1", "years": [1990], "geos": ["nation", "state", "county", "tract"]},
    {"label": "Decennial 2000 SF1", "slug": "dec/sf1", "years": [2000], "geos": ["nation", "state", "county", "tract"]},
    {"label": "Decennial 2010 SF1", "slug": "dec/sf1", "years": [2010], "geos": ["nation", "state", "county", "tract"]},
    {"label": "Decennial 2020 PL (Redistricting)", "slug": "dec/pl", "years": [2020], "geos": ["nation", "state", "county", "tract"]},
    {"label": "Population Estimates (PEP) — Total Pop", "slug": "pep/population", "years": list(range(2010, 2024)), "geos": ["nation", "state", "county"]},
]
DEFAULT_VARS_BY_SLUG: Dict[str, List[str]] = {
    "acs/acs1": ["B01003_001E"],   # Total population
    "acs/acs5": ["B01003_001E"],
    "dec/pl":  ["P1_001N"],        # 2020 PL total pop
    "dec/sf1": ["P001001"],        # 2010/2000/1990 SF1 total pop
    "pep/population": ["POP"],     # PEP total pop
}

# ---------------------------
# Cached helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_variables(dataset_slug: str, year: int) -> pd.DataFrame:
    """Return variables.json as a DataFrame with helpful flags."""
    url = f"https://api.census.gov/data/{year}/{dataset_slug}/variables.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    rows = []
    for name, meta in data.get("variables", {}).items():
        rows.append({
            "name": name,
            "label": meta.get("label", ""),
            "concept": meta.get("concept", ""),
            "is_estimate_E": str(name).endswith("E"),
        })
    df = pd.DataFrame(rows)
    return df

@st.cache_data(show_spinner=False)
def fetch_states(dataset_slug: str, year: int) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/{dataset_slug}"
    params = {"get": "NAME", "for": "state:*"}
    if API_KEY:
        params["key"] = API_KEY
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data[1:], columns=data[0]).rename(columns={"state": "state_fips"})

@st.cache_data(show_spinner=False)
def fetch_counties(dataset_slug: str, year: int, state_fips: str) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/{dataset_slug}"
    params = {"get": "NAME", "for": "county:*", "in": f"state:{state_fips}"}
    if API_KEY:
        params["key"] = API_KEY
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return (
        pd.DataFrame(data[1:], columns=data[0])
        .rename(columns={"county": "county_fips", "state": "state_fips"})
    )

@st.cache_data(show_spinner=False)
def census_query(
    dataset_slug: str,
    year: int,
    variables: List[str],
    geo_level: str,
    state_fips: Optional[str] = None,
    county_fips: Optional[str] = None,
) -> pd.DataFrame:
    """Query the Census API and return results as DataFrame."""
    base = f"https://api.census.gov/data/{year}/{dataset_slug}"

    # Build geo clauses
    if geo_level == "nation":
        for_clause, in_clause = "us:1", None
    elif geo_level == "state":
        if state_fips and state_fips != "*":
            for_clause, in_clause = f"state:{state_fips}", None
        else:
            for_clause, in_clause = "state:*", None
    elif geo_level == "county":
        if not state_fips:
            raise ValueError("State is required for county queries.")
        if county_fips and county_fips != "*":
            for_clause, in_clause = f"county:{county_fips}", f"state:{state_fips}"
        else:
            for_clause, in_clause = "county:*", f"state:{state_fips}"
    elif geo_level == "tract":
        if not (state_fips and county_fips):
            raise ValueError("State and County are required for tract queries.")
        for_clause, in_clause = "tract:*", f"state:{state_fips}+county:{county_fips}"
    else:
        raise ValueError("Unsupported geo level")

    get_vars = ["NAME"] + (variables or [])
    params = {"get": ",".join(get_vars), "for": for_clause}
    if in_clause:
        params["in"] = in_clause
    if API_KEY:
        params["key"] = API_KEY

    url = base + "?" + urlencode(params)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    # Coerce numerics where possible
    for c in df.columns:
        if c not in {"NAME", "state", "county", "tract"}:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # Compose GEOID
    if geo_level == "nation":
        df["GEOID"] = "0100000US"
    elif geo_level == "state":
        df["GEOID"] = "0400000US" + df["state"].astype(str).str.zfill(2)
    elif geo_level == "county":
        df["GEOID"] = "0500000US" + df["state"].astype(str).str.zfill(2) + df["county"].astype(str).str.zfill(3)
    elif geo_level == "tract":
        df["GEOID"] = (
            "1400000US"
            + df["state"].astype(str).str.zfill(2)
            + df["county"].astype(str).str.zfill(3)
            + df["tract"].astype(str).str.zfill(6)
        )
    return df

# ---------------------------
# UI (no separate search input)
# ---------------------------
st.title("Census Data Dashboard")

left, right = st.columns([2.2, 1.0], gap="large")

with left:
    ds_label = st.selectbox("Dataset", options=[d["label"] for d in DATASETS], index=1)
    ds = next(d for d in DATASETS if d["label"] == ds_label)

    year = st.selectbox("Year", options=sorted(ds["years"], reverse=True))
    geo = st.selectbox("Geography", options=ds["geos"], index=1)

    state_fips: Optional[str] = None
    county_fips: Optional[str] = None

    if geo in ("state", "county", "tract"):
        states_df = fetch_states(ds["slug"], year)
        state_choices = ["All States (*)"] + states_df["NAME"].tolist()
        default_state_index = 0
        if "Texas" in states_df["NAME"].values:
            default_state_index = state_choices.index("Texas")
        state_choice = st.selectbox("State", options=state_choices, index=max(0, default_state_index))
        state_fips = "*" if state_choice == "All States (*)" else states_df.loc[states_df["NAME"] == state_choice, "state_fips"].iloc[0]

    if geo in ("county", "tract") and state_fips and state_fips != "*":
        counties_df = fetch_counties(ds["slug"], year, state_fips)
        county_choices = ["All Counties (*)"] + counties_df["NAME"].tolist()
        county_choice = st.selectbox("County", options=county_choices, index=0)
        county_fips = "*" if county_choice == "All Counties (*)" else counties_df.loc[counties_df["NAME"] == county_choice, "county_fips"].iloc[0]

with right:
    st.markdown("**API key embedded for all users:** ✅")
    st.caption("Tip: Use the type-to-filter box inside the *Variables* dropdown to find items (e.g., “median income”, “poverty”, “B01003”).")

# Variables (full list; multiselect has built-in search)
vars_df = fetch_variables(ds["slug"], year).sort_values(["is_estimate_E", "name"], ascending=[False, True])
choices = [f"{r.name} — {r.label}" for r in vars_df.itertuples(index=False)]
choice_to_var = dict(zip(choices, vars_df["name"]))

defaults = DEFAULT_VARS_BY_SLUG.get(ds["slug"], [])
default_labels = [f"{v} — {vars_df.loc[vars_df['name'] == v, 'label'].iloc[0]}"
                  for v in defaults if v in vars_df["name"].values]
if not default_labels and choices:
    default_labels = [choices[0]]

selected_labels = st.multiselect("Variables", options=choices, default=default_labels, max_selections=80)
selected_vars = [choice_to_var[c] for c in selected_labels]

# Fetch button
fetch = st.button("Fetch data", type="primary")

if fetch:
    if not selected_vars:
        st.warning("Select at least one variable.")
        st.stop()

    try:
        df = census_query(ds["slug"], year, selected_vars, geo, state_fips, county_fips)
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

    st.success(f"Got {len(df):,} rows.")

    value_cols = [c for c in df.columns if c not in ("NAME", "GEOID", "state", "county", "tract")]
    ordered_cols = ["NAME"] + value_cols + [c for c in ("state", "county", "tract", "GEOID") if c in df.columns]
    df_view = df[ordered_cols]

    st.dataframe(df_view, use_container_width=True, height=420)

    if value_cols and geo in ("state", "county") and len(df_view) > 1:
        metric = value_cols[0]
        plot_df = df_view.nlargest(50, metric).copy() if len(df_view) > 80 else df_view.copy()
        fig = px.bar(plot_df, x="NAME", y=metric, title=f"Top {len(plot_df)} by {metric}")
        fig.update_layout(xaxis_title="", yaxis_title=metric, xaxis_tickangle=-45, height=480)
        st.plotly_chart(fig, use_container_width=True)

    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    fname = f"census_{ds['slug'].replace('/', '_')}_{year}_{geo}.csv"
    st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")

st.markdown(
    """
---
**Notes**
- Years & geographies update automatically based on the dataset chosen.
- County selector appears only when relevant (Geo = county or tract; requires selecting a State).
"""
)
