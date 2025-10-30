# app.py — Census Data Dashboard with Variable Groups
# Streamlit + US Census API

import requests
import pandas as pd
import streamlit as st
from functools import lru_cache

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="Census Data Dashboard — by SHAH", layout="wide")
st.title("Census Data Dashboard — by Shah")

# Put this near the top of app.py
def clear_all_caches():
    try:
        get_years.cache_clear()
        get_groups.cache_clear()
        get_group_variables.cache_clear()
        get_states.cache_clear()
        get_counties.cache_clear()
    except Exception:
        pass

with st.sidebar:
    if st.button("🔄 Force refresh (clear caches)"):
        clear_all_caches()
        st.cache_data.clear()       # if you use st.cache_data anywhere
        st.experimental_rerun()


# ------------------------
# Configuration
# ------------------------
API_KEY = "6c14346c155c7ae9110a833a854582dc60c3afd0"   # embedded so users don't need one

# Datasets you want to expose in the UI
DATASETS = {
    "ACS 1-year": "acs/acs1",
    "ACS 5-year": "acs/acs5",
    "Decennial 2010 SF1": "dec/sf1",
}

# A tiny map to make some common codes read nicely in the table header
COMMON_CODE_NAMES = {
    "NAME": "Name",
    "state": "State FIPS",
    "county": "County FIPS",
    "tract": "Tract FIPS",
    "GEOID": "GEOID",
}

# ------------------------
# Helpers (cached)
# ------------------------
def api_base(year: int, dataset_path: str) -> str:
    return f"https://api.census.gov/data/{year}/{dataset_path}"

@lru_cache(maxsize=128)
def get_years(dataset_path: str) -> list[int]:
    """Return available years that actually exist for this dataset."""
    # We discover years by probing the API /variables endpoint backwards from 2025.
    candidates = list(range(2025, 2009, -1))  # 2025..2010
    exists = []
    for y in candidates:
        url = f"{api_base(y, dataset_path)}/variables.json"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            exists.append(y)
    exists.sort(reverse=True)
    return exists

@lru_cache(maxsize=128)
def get_groups(year: int, dataset_path: str) -> pd.DataFrame:
    """Return the groups (code + description) for a dataset/year."""
    url = f"{api_base(year, dataset_path)}/groups.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    # groups is a dict keyed by group name with description, name, variables, etc.
    rows = []
    for gcode, ginfo in js.get("groups", {}).items():
        rows.append({
            "group_code": gcode,                         # e.g., "B01001"
            "name": ginfo.get("name", gcode),            # e.g., "SEX BY AGE"
            "description": ginfo.get("description", ""), # often nicer sentence
        })
    df = pd.DataFrame(rows).sort_values("group_code").reset_index(drop=True)
    return df

@lru_cache(maxsize=256)
def get_group_variables(year: int, dataset_path: str, group_code: str) -> pd.DataFrame:
    """Return the variables within a specific group."""
    url = f"{api_base(year, dataset_path)}/groups/{group_code}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    vars_dict = js.get("variables", {})
    rows = []
    for vid, vinfo in vars_dict.items():
        label = vinfo.get("label", "")
        concept = vinfo.get("concept", "")
        if vid.endswith("EA"):  # avoid annotations; keep *_E/_M/main only
            continue
        rows.append({
            "id": vid,
            "label": label,
            "concept": concept,
        })
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    return df

@lru_cache(maxsize=256)
def get_states(year: int, dataset_path: str) -> pd.DataFrame:
    url = f"{api_base(year, dataset_path)}?get=NAME&for=state:*"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["state_name"] = df["NAME"]
    df["state_fips"] = df["state"]
    return df[["state_name", "state_fips"]].sort_values("state_name")

@lru_cache(maxsize=256)
def get_counties(year: int, dataset_path: str, state_fips: str) -> pd.DataFrame:
    url = f"{api_base(year, dataset_path)}?get=NAME&for=county:*&in=state:{state_fips}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["county_name"] = df["NAME"]
    df["county_fips"] = df["county"]
    return df[["county_name", "county_fips"]].sort_values("county_name")

def friendly_header(col: str) -> str:
    return COMMON_CODE_NAMES.get(col, col)

def prettify_label(vrow) -> str:
    """Make the variable label shorter but readable."""
    label = vrow["label"]
    concept = vrow["concept"]
    # Replace weird '!!' patterns with arrows and collapse extra bangs
    short = label.replace("!!", " → ").replace("  ", " ")
    return f'{vrow["id"]} — {short}'

# ------------------------
# Query builder & fetch
# ------------------------
def build_where(geo: str, state_fips: str | None, county_fips: str | None):
    if geo == "nation":
        return {"for": "us:1"}, None
    if geo == "state":
        if state_fips and state_fips != "*":
            return {"for": f"state:{state_fips}"}, None
        else:
            return {"for": "state:*"}, None
    if geo == "county":
        if not state_fips or state_fips == "*":
            raise ValueError("Select a State for county geography.")
        if county_fips and county_fips != "*":
            return {"for": f"county:{county_fips}", "in": f"state:{state_fips}"}, None
        else:
            return {"for": "county:*", "in": f"state:{state_fips}"}, None
    if geo == "tract":
        if not state_fips or not county_fips or state_fips == "*" or county_fips == "*":
            raise ValueError("Select a specific State and County for tract geography.")
        return {"for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}, None
    raise ValueError("Unsupported geography.")

def census_fetch(year: int,
                 dataset_path: str,
                 variables: list[str],
                 geo: str,
                 state_fips: str | None,
                 county_fips: str | None) -> pd.DataFrame:
    base = api_base(year, dataset_path)
    params, _ = build_where(geo, state_fips, county_fips)
    var_list = ",".join(["NAME"] + variables)
    params = {"get": var_list, **(params or {}), "key": API_KEY}
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    # Add GEOID where appropriate
    if "state" in df.columns and "county" in df.columns and "tract" in df.columns:
        df["GEOID"] = df["state"] + df["county"] + df["tract"]
    elif "state" in df.columns and "county" in df.columns:
        df["GEOID"] = df["state"] + df["county"]
    elif "state" in df.columns:
        df["GEOID"] = df["state"]
    # Order columns
    leading = ["NAME"] + [c for c in ["state", "county", "tract", "GEOID"] if c in df.columns]
    rest = [c for c in df.columns if c not in leading]
    df = df[leading + rest]
    df = df.rename(columns={c: friendly_header(c) for c in df.columns})
    return df

# ------------------------
# Sidebar / Controls
# ------------------------
colA, colB = st.columns([1, 1])

with colA:
    dataset_label = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    dataset_path = DATASETS[dataset_label]

    years = get_years(dataset_path)
    if not years:
        st.stop()
    year = st.selectbox("Year", years, index=0)

    geo = st.selectbox("Geography", ["nation", "state", "county", "tract"], index=1)

with colB:
    # State & County controls (show only when relevant)
    state_fips = None
    county_fips = None

    if geo in ("state", "county", "tract"):
        states_df = get_states(year, dataset_path)
        state_display = ["All States (*)"] + [f'{r.state_name} ({r.state_fips})' for r in states_df.itertuples()]
        chosen_state = st.selectbox("State", state_display, index=0)
        if chosen_state.startswith("All States"):
            state_fips = "*"
        else:
            state_fips = chosen_state.split("(")[-1].strip(")")

    if geo in ("county", "tract") and state_fips and state_fips != "*":
        counties_df = get_counties(year, dataset_path, state_fips)
        county_display = ["All Counties (*)"] + [f'{r.county_name} ({r.county_fips})' for r in counties_df.itertuples()]
        chosen_county = st.selectbox("County", county_display, index=0)
        if chosen_county.startswith("All Counties"):
            county_fips = "*"
        else:
            county_fips = chosen_county.split("(")[-1].strip(")")

# --- Group → Variables cascade ---
groups_df = get_groups(year, dataset_path)
group_options = [f'{r.group_code} — {r.description or r.name}' for r in groups_df.itertuples()]
st.markdown("#### Variables")
group_choice = st.selectbox("Variable group", group_options, index=0, help="Pick a group, then choose variables from that group.")

selected_group = group_choice.split(" — ")[0] if group_choice else None

vars_df = pd.DataFrame()
if selected_group:
    vars_df = get_group_variables(year, dataset_path, selected_group)
    # Pretty labels for UI
    vars_df["pretty"] = vars_df.apply(prettify_label, axis=1)

selected_vars_pretty = st.multiselect(
    "Variables",
    options=list(vars_df["pretty"]) if not vars_df.empty else [],
    max_selections=80,
    help="Type to filter within this group."
)

selected_var_ids = list(vars_df.loc[vars_df["pretty"].isin(selected_vars_pretty), "id"]) if not vars_df.empty else []

# Link to the group's API page (handy)
if selected_group:
    st.caption(
        f"[Open Census API group page for {selected_group}]({api_base(year, dataset_path)}/groups/{selected_group}.json)"
    )

# ------------------------
# Fetch
# ------------------------
if st.button("Fetch data", type="primary"):
    if not selected_var_ids:
        st.warning("Pick at least one variable.")
        st.stop()
    try:
        df = census_fetch(year, dataset_path, selected_var_ids, geo, state_fips, county_fips)
        st.success(f"Got {len(df):,} rows.")
        st.dataframe(df, use_container_width=True, height=480)

        # simple profile chart when possible
        if geo in ("state", "county") and len(selected_var_ids) == 1 and len(df) > 1:
            import plotly.express as px
            metric = selected_var_ids[0]
            fig = px.bar(
                df,
                x="Name",
                y=metric,
                title=f"{metric} by {geo}",
                labels={"Name": geo.capitalize()},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        fname = f"{dataset_label.replace(' ', '_')}_{year}_{geo}.csv"
        st.download_button("Download CSV", data=csv, file_name=fname, mime="text/csv")

    except Exception as e:
        st.error(f"Fetch failed: {e}")

# ------------------------
# Notes
# ------------------------
st.markdown("""
**Notes**
- Years & geographies update automatically based on the dataset chosen.
- For **county** and **tract**, pick a **State** first (and a **County** for tracts).
- Variable selection is now **grouped**—choose a **group** first to declutter the list.
""")
