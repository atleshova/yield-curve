
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transparent Nelson-Siegel App", layout="wide")

# Exact FRED maturity mapping.
# Important: codes are matched exactly, so DGS30 is not confused with DGS3.
MATURITY_MAP = {
    "DGS3MO": 0.25,
    "DGS6MO": 0.50,
    "DGS1": 1.00,
    "DGS2": 2.00,
    "DGS3": 3.00,
    "DGS5": 5.00,
    "DGS7": 7.00,
    "DGS10": 10.00,
    "DGS20": 20.00,
    "DGS30": 30.00,
}

ORDERED_CODES = sorted(MATURITY_MAP.keys(), key=len, reverse=True)


def detect_fred_code(filename, columns):
    """
    Detect the FRED code safely.
    Fixes the earlier bug where DGS30 was accidentally detected as DGS3.
    """
    name = filename.upper()

    # Exact filename match: DGS30.xlsx -> DGS30
    stem = re.sub(r"\.(CSV|XLSX|XLS)$", "", name)
    if stem in MATURITY_MAP:
        return stem

    # Search exact code tokens in filename
    for code in ORDERED_CODES:
        pattern = rf"(^|[^A-Z0-9]){code}([^A-Z0-9]|$)"
        if re.search(pattern, name):
            return code

    # Search exact column names
    clean_cols = [str(c).strip().upper() for c in columns]
    for code in ORDERED_CODES:
        if code in clean_cols:
            return code

    return None


def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name

    if filename.upper().endswith(".CSV"):
        df = pd.read_csv(uploaded_file)
        sheet_used = "CSV"
    else:
        xls = pd.ExcelFile(uploaded_file)
        if "Daily" in xls.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name="Daily")
            sheet_used = "Daily"
        else:
            df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
            sheet_used = xls.sheet_names[0]

    df.columns = [str(c).strip() for c in df.columns]
    fred_code = detect_fred_code(filename, df.columns)

    if fred_code is None:
        raise ValueError(f"Could not detect FRED code from file name or columns: {filename}")

    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["observation_date", "date"]:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    if fred_code in df.columns:
        yield_col = fred_code
    else:
        possible_numeric = [c for c in df.columns if c != date_col]
        yield_col = None
        for c in possible_numeric:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                yield_col = c
                break
        if yield_col is None:
            raise ValueError(f"Could not find numeric yield column in {filename}")

    out = df[[date_col, yield_col]].copy()
    out.columns = ["Date", fred_code]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[fred_code] = pd.to_numeric(out[fred_code], errors="coerce")
    out = out.dropna(subset=["Date", fred_code])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"])

    info = {
        "file": filename,
        "sheet_used": sheet_used,
        "detected_code": fred_code,
        "maturity_years": MATURITY_MAP[fred_code],
        "rows_read": len(out),
        "first_date": out["Date"].min(),
        "last_date": out["Date"].max(),
    }

    return out, info


def merge_files(uploaded_files):
    dfs = []
    info_rows = []
    seen = set()

    for f in uploaded_files:
        df, info = read_uploaded_file(f)
        code = info["detected_code"]

        if code in seen:
            raise ValueError(
                f"Duplicate series detected: {code}. Upload only one file for each maturity."
            )
        seen.add(code)

        dfs.append(df)
        info_rows.append(info)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Date", how="inner")

    yield_cols = [c for c in merged.columns if c != "Date"]
    yield_cols = sorted(yield_cols, key=lambda c: MATURITY_MAP[c])
    merged = merged[["Date"] + yield_cols].sort_values("Date").reset_index(drop=True)

    return merged, pd.DataFrame(info_rows), yield_cols


def monthly_last_observation(df):
    temp = df.copy()
    temp["Month"] = temp["Date"].dt.to_period("M")
    out = temp.sort_values("Date").groupby("Month").tail(1)
    return out.drop(columns=["Month"]).reset_index(drop=True)


def ns_loadings(maturities, lambd):
    tau = np.asarray(maturities, dtype=float)
    x0 = np.ones_like(tau)
    x1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    x2 = x1 - np.exp(-lambd * tau)
    return np.column_stack([x0, x1, x2])


def nelson_siegel_yield(maturities, beta0, beta1, beta2, lambd):
    X = ns_loadings(maturities, lambd)
    return X @ np.array([beta0, beta1, beta2])


def fit_ns_for_lambda(yields, maturities, lambd):
    y = np.asarray(yields, dtype=float)
    X = ns_loadings(maturities, lambd)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ betas
    error = y - fitted
    sse = float(np.sum(error ** 2))
    return betas, fitted, error, sse, X


def fit_ns_grid(yields, maturities, lambda_grid):
    rows = []
    best = None

    for lambd in lambda_grid:
        betas, fitted, error, sse, X = fit_ns_for_lambda(yields, maturities, lambd)
        row = {
            "lambda": float(lambd),
            "beta0_level": float(betas[0]),
            "beta1_slope": float(betas[1]),
            "beta2_curvature": float(betas[2]),
            "SSE": sse,
        }
        rows.append(row)
        if best is None or sse < best["SSE"]:
            best = row

    return best, pd.DataFrame(rows)


def estimate_all_dates(model_data, yield_cols, lambda_grid):
    maturities = [MATURITY_MAP[c] for c in yield_cols]
    results = []
    total = len(model_data)
    progress = st.progress(0)

    for i, (_, row) in enumerate(model_data.iterrows()):
        yields = row[yield_cols].to_numpy(dtype=float)
        best, _ = fit_ns_grid(yields, maturities, lambda_grid)
        results.append({"Date": row["Date"], **best})
        progress.progress((i + 1) / total)

    return pd.DataFrame(results)


def flag_results(best, lambda_min, lambda_max):
    warnings = []

    if abs(best["lambda"] - lambda_min) < 1e-12:
        warnings.append("Best lambda is at the minimum boundary. Try lowering lambda minimum or check maturity mapping.")
    if abs(best["lambda"] - lambda_max) < 1e-12:
        warnings.append("Best lambda is at the maximum boundary. Try raising lambda maximum.")
    if max(abs(best["beta0_level"]), abs(best["beta1_slope"]), abs(best["beta2_curvature"])) > 100:
        warnings.append("One or more beta values are extremely large. This usually means the curve is poorly identified, too few maturities are used, or lambda range is problematic.")

    return warnings


st.title("Transparent Nelson-Siegel Yield Curve App")

st.markdown(
    """
This version is designed so you can **see what the model is doing**, instead of treating it like a black box.

The Nelson-Siegel model fitted here is:

$$
y(\\tau)=\\beta_0+\\beta_1\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau}\\right)
+\\beta_2\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau}-e^{-\\lambda\\tau}\\right)
$$
"""
)

with st.sidebar:
    st.header("Settings")
    frequency = st.radio("Use data frequency", ["Monthly last observation", "Daily"], index=0)
    lambda_min = st.number_input("Lambda minimum", min_value=0.001, value=0.01, step=0.01, format="%.3f")
    lambda_max = st.number_input("Lambda maximum", min_value=0.01, value=5.00, step=0.10, format="%.3f")
    lambda_step = st.number_input("Lambda grid step", min_value=0.001, value=0.01, step=0.005, format="%.3f")
    forecast_horizon = st.number_input("Forecast horizon, rows/months", min_value=1, value=1, step=1)

st.info(
    "Upload FRED files such as DGS3MO.xlsx, DGS6MO.xlsx, DGS1.xlsx, DGS2.xlsx, DGS10.xlsx, DGS30.xlsx. "
    "The app reads the FRED `Daily` sheet automatically when it exists."
)

uploaded_files = st.file_uploader(
    "Upload Treasury yield files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()

try:
    merged, file_info, yield_cols = merge_files(uploaded_files)
except Exception as e:
    st.error(str(e))
    st.stop()

st.header("1. File detection and maturity mapping")

st.write("This table shows what the app thinks each uploaded file represents.")
st.dataframe(file_info, use_container_width=True)

mapping_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity in years": [MATURITY_MAP[c] for c in yield_cols],
})
st.dataframe(mapping_df, use_container_width=True)

if len(yield_cols) < 4:
    st.warning("Use at least 4 maturities. Six or more is better. Nelson-Siegel has three beta parameters plus lambda.")

st.header("2. Merged yield data")

st.write("Only dates common to all uploaded files are kept.")
st.write(f"Rows after merge: {len(merged)}")
st.dataframe(merged.head(25), use_container_width=True)

model_data = monthly_last_observation(merged) if frequency == "Monthly last observation" else merged.copy()

st.header("3. Data used for estimation")
st.write(f"Frequency selected: **{frequency}**")
st.write(f"Rows used for estimation: {len(model_data)}")
st.dataframe(model_data.head(25), use_container_width=True)

st.header("4. Single-date calculation, fully visible")

date_options = model_data["Date"].dt.strftime("%Y-%m-%d").tolist()
selected_date = st.selectbox("Choose date", date_options, index=len(date_options) - 1)

row = model_data[model_data["Date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
maturities = [MATURITY_MAP[c] for c in yield_cols]
actual_yields = row[yield_cols].to_numpy(dtype=float)

single_data = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity tau": maturities,
    "Actual yield": actual_yields,
})
st.subheader("4A. Actual curve for selected date")
st.dataframe(single_data, use_container_width=True)

lambda_grid = np.arange(lambda_min, lambda_max + lambda_step / 2, lambda_step)
best, grid_table = fit_ns_grid(actual_yields, maturities, lambda_grid)

warnings = flag_results(best, lambda_min, lambda_max)
for w in warnings:
    st.warning(w)

st.subheader("4B. Best grid-search result")

cols = st.columns(5)
cols[0].metric("Best lambda", f"{best['lambda']:.4f}")
cols[1].metric("beta0 level", f"{best['beta0_level']:.4f}")
cols[2].metric("beta1 slope", f"{best['beta1_slope']:.4f}")
cols[3].metric("beta2 curvature", f"{best['beta2_curvature']:.4f}")
cols[4].metric("SSE", f"{best['SSE']:.6f}")

betas, fitted, errors, sse, X = fit_ns_for_lambda(actual_yields, maturities, best["lambda"])

loading_df = pd.DataFrame({
    "FRED code": yield_cols,
    "tau": maturities,
    "X0 beta0 loading": X[:, 0],
    "X1 beta1 loading": X[:, 1],
    "X2 beta2 loading": X[:, 2],
})
st.subheader("4C. Nelson-Siegel loadings")
st.write("These are the equivalent of the Excel formula columns.")
st.dataframe(loading_df, use_container_width=True)

fit_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity tau": maturities,
    "Actual yield": actual_yields,
    "Fitted yield": fitted,
    "Error actual - fitted": errors,
    "Squared error": errors ** 2,
})
st.subheader("4D. Fitted values and errors")
st.dataframe(fit_df, use_container_width=True)

st.markdown(
    """
**What happened here?**

1. The app tried many possible lambda values.
2. For each lambda, it computed the two Nelson-Siegel loading columns.
3. For that fixed lambda, beta0, beta1, and beta2 were estimated using ordinary least squares.
4. The app calculated SSE, the sum of squared errors.
5. The lambda with the smallest SSE was selected.
"""
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(maturities, actual_yields, marker="o", label="Actual")
ax.plot(maturities, fitted, marker="o", label="Nelson-Siegel fitted")
ax.set_xlabel("Maturity in years")
ax.set_ylabel("Yield, percent")
ax.set_title(f"Actual vs fitted yield curve: {selected_date}")
ax.legend()
st.pyplot(fig)

with st.expander("Show full lambda grid search table"):
    st.dataframe(grid_table, use_container_width=True)

st.header("5. Estimate factors for every date")

if st.button("Run estimation for all dates"):
    with st.spinner("Running Nelson-Siegel grid search for all dates..."):
        factors = estimate_all_dates(model_data, yield_cols, lambda_grid)
        st.session_state["factors"] = factors

if "factors" in st.session_state:
    factors = st.session_state["factors"]

    st.subheader("Estimated factor table")
    st.dataframe(factors, use_container_width=True)

    st.subheader("Factor chart")
    chart_cols = ["beta0_level", "beta1_slope", "beta2_curvature", "lambda"]
    st.line_chart(factors.set_index("Date")[chart_cols])

    if (factors["lambda"] == lambda_min).mean() > 0.5:
        st.warning("More than half of fitted dates choose the minimum lambda. Check whether your lambda range is too high or whether too few maturities are uploaded.")

    st.download_button(
        "Download Nelson-Siegel factors CSV",
        data=factors.to_csv(index=False).encode("utf-8"),
        file_name="nelson_siegel_factors.csv",
        mime="text/csv",
    )

    st.header("6. Forecasting dataset")
    forecast = factors.copy()
    forecast["future_beta0_level"] = forecast["beta0_level"].shift(-forecast_horizon)
    forecast["future_beta1_slope"] = forecast["beta1_slope"].shift(-forecast_horizon)
    forecast["future_beta2_curvature"] = forecast["beta2_curvature"].shift(-forecast_horizon)
    forecast = forecast.dropna().reset_index(drop=True)

    st.write("This is the dataset you will later use for regression / random forest / XGBoost forecasting.")
    st.dataframe(forecast, use_container_width=True)

    st.download_button(
        "Download forecasting dataset CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="forecast_dataset.csv",
        mime="text/csv",
    )
