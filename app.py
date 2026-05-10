
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nelson-Siegel Yield Curve Research App", layout="wide")

MATURITY_MAP = {
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.upper()

    if filename.endswith(".CSV"):
        df = pd.read_csv(uploaded_file)
else:
    xls = pd.ExcelFile(uploaded_file)

    if "Daily" in xls.sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name="Daily")
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    for c in df.columns:
        if c.lower() in ["observation_date", "date"]:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    series_name = None
    for code in MATURITY_MAP:
        if code in filename:
            series_name = code
            break

    if series_name is None:
        for c in df.columns:
            if c in MATURITY_MAP:
                series_name = c
                break

    if series_name is not None and series_name in df.columns:
        series_col = series_name
    else:
        series_col = None
        for c in df.columns:
            if c != "Date":
                numeric_test = pd.to_numeric(df[c], errors="coerce")
                if numeric_test.notna().sum() > 0:
                    series_col = c
                    break
        if series_name is None:
            series_name = str(series_col)

    if series_col is None:
        raise ValueError(f"Could not identify yield column in {uploaded_file.name}")

    out = df[["Date", series_col]].copy()
    out[series_col] = pd.to_numeric(out[series_col], errors="coerce")
    out = out.dropna()

    if series_col != series_name:
        out = out.rename(columns={series_col: series_name})

    return out, series_name

def merge_files(uploaded_files):
    merged = None
    for f in uploaded_files:
        df, _ = read_uploaded_file(f)
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)

def ns_loadings(maturities, lambd):
    tau = np.asarray(maturities, dtype=float)
    x0 = np.ones_like(tau)
    x1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    x2 = x1 - np.exp(-lambd * tau)
    return np.column_stack([x0, x1, x2])

def nelson_siegel_yield(maturities, beta0, beta1, beta2, lambd):
    X = ns_loadings(maturities, lambd)
    return X @ np.array([beta0, beta1, beta2])

def fit_ns_grid(yields, maturities, lambda_grid):
    y = np.asarray(yields, dtype=float)
    best = None
    rows = []

    for lambd in lambda_grid:
        X = ns_loadings(maturities, lambd)
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ betas
        sse = float(np.sum((y - fitted) ** 2))
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

def monthly_last_observation(df):
    temp = df.copy()
    temp["YearMonth"] = temp["Date"].dt.to_period("M")
    monthly = temp.sort_values("Date").groupby("YearMonth").tail(1)
    return monthly.drop(columns=["YearMonth"]).reset_index(drop=True)

def estimate_factors_for_all_dates(df, yield_cols, lambda_min, lambda_max, lambda_step):
    maturities = [MATURITY_MAP[c] for c in yield_cols]
    lambda_grid = np.arange(lambda_min, lambda_max + lambda_step / 2, lambda_step)
    results = []

    progress = st.progress(0)
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        y = row[yield_cols].values.astype(float)
        best, _ = fit_ns_grid(y, maturities, lambda_grid)
        results.append({"Date": row["Date"], **best})
        progress.progress((i + 1) / total)

    return pd.DataFrame(results)

def make_forecast_dataset(factors, horizon_months):
    out = factors.copy()
    out["future_beta0_level"] = out["beta0_level"].shift(-horizon_months)
    out["future_beta1_slope"] = out["beta1_slope"].shift(-horizon_months)
    out["future_beta2_curvature"] = out["beta2_curvature"].shift(-horizon_months)
    return out.dropna().reset_index(drop=True)

st.title("Nelson-Siegel Yield Curve Research App")

st.write(
    "Upload FRED Treasury yield CSV/XLSX files, estimate real Nelson-Siegel parameters "
    "using grid search, visualize fitted curves, and export factor datasets."
)

with st.sidebar:
    st.header("Settings")
    frequency = st.radio("Frequency", ["Monthly last observation", "Daily"], index=0)
    lambda_min = st.number_input("Lambda minimum", min_value=0.001, value=0.01, step=0.01, format="%.3f")
    lambda_max = st.number_input("Lambda maximum", min_value=0.01, value=5.00, step=0.10, format="%.3f")
    lambda_step = st.number_input("Lambda grid step", min_value=0.001, value=0.01, step=0.005, format="%.3f")
    horizon_months = st.number_input("Forecast horizon", min_value=1, value=1, step=1)

uploaded_files = st.file_uploader(
    "Upload FRED files such as DGS3MO.xlsx, DGS6MO.xlsx, DGS1.xlsx, DGS2.xlsx, DGS10.xlsx, DGS30.xlsx",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least four Treasury maturity files to begin.")
    st.stop()

try:
    merged = merge_files(uploaded_files)
except Exception as e:
    st.error(f"Could not read uploaded files: {e}")
    st.stop()

yield_cols = [c for c in merged.columns if c != "Date" and c in MATURITY_MAP]
yield_cols = sorted(yield_cols, key=lambda c: MATURITY_MAP[c])

if len(yield_cols) < 4:
    st.warning("Nelson-Siegel works better with at least four maturities. Upload more files if possible.")

st.header("1. Merged Yield Data")
st.write(f"Detected series: {', '.join(yield_cols)}")
st.dataframe(merged, use_container_width=True)

model_data = monthly_last_observation(merged) if frequency == "Monthly last observation" else merged.copy()

st.subheader("Data used for estimation")
st.write(f"Rows used: {len(model_data)}")
st.dataframe(model_data.head(20), use_container_width=True)

st.header("2. Single-Date Nelson-Siegel Fit")

selected_date = st.selectbox(
    "Choose a date",
    options=model_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
    index=len(model_data) - 1,
)

date_row = model_data[model_data["Date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
maturities = [MATURITY_MAP[c] for c in yield_cols]
yields = date_row[yield_cols].values.astype(float)
lambda_grid = np.arange(lambda_min, lambda_max + lambda_step / 2, lambda_step)

best, grid_table = fit_ns_grid(yields, maturities, lambda_grid)
fitted = nelson_siegel_yield(
    maturities,
    best["beta0_level"],
    best["beta1_slope"],
    best["beta2_curvature"],
    best["lambda"],
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("β0 Level", f"{best['beta0_level']:.4f}")
col2.metric("β1 Slope", f"{best['beta1_slope']:.4f}")
col3.metric("β2 Curvature", f"{best['beta2_curvature']:.4f}")
col4.metric("Best λ", f"{best['lambda']:.4f}")

fit_df = pd.DataFrame({
    "Maturity": maturities,
    "Actual Yield": yields,
    "Fitted Yield": fitted,
    "Error": yields - fitted,
})
st.dataframe(fit_df, use_container_width=True)

fig, ax = plt.subplots()
ax.plot(maturities, yields, marker="o", label="Actual")
ax.plot(maturities, fitted, marker="o", label="Nelson-Siegel fitted")
ax.set_xlabel("Maturity in years")
ax.set_ylabel("Yield")
ax.set_title(f"Nelson-Siegel Fit on {selected_date}")
ax.legend()
st.pyplot(fig)

with st.expander("Show lambda grid search table"):
    st.dataframe(grid_table, use_container_width=True)

st.header("3. Estimate Nelson-Siegel Factors for All Dates")

if st.button("Run factor estimation"):
    with st.spinner("Estimating factors..."):
        st.session_state["factors"] = estimate_factors_for_all_dates(
            model_data, yield_cols, lambda_min, lambda_max, lambda_step
        )

if "factors" in st.session_state:
    factors = st.session_state["factors"]
    st.subheader("Estimated Nelson-Siegel Factors")
    st.dataframe(factors, use_container_width=True)

    st.line_chart(factors.set_index("Date")[["beta0_level", "beta1_slope", "beta2_curvature", "lambda"]])

    st.download_button(
        "Download Nelson-Siegel factors CSV",
        data=factors.to_csv(index=False).encode("utf-8"),
        file_name="nelson_siegel_factors.csv",
        mime="text/csv",
    )

    st.header("4. Forecast Dataset")
    forecast_df = make_forecast_dataset(factors, horizon_months)
    st.dataframe(forecast_df, use_container_width=True)

    st.download_button(
        "Download forecasting dataset CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_dataset.csv",
        mime="text/csv",
    )

st.header("5. Explanation")
st.markdown(
    """
    **Grid search** means the app tries many possible λ values.  
    For each λ, it estimates β0, β1, and β2 by ordinary least squares.  
    Then it chooses the λ with the lowest sum of squared errors.

    This estimates the real Nelson-Siegel model without using Excel Solver.
    """
)
