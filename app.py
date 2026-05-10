
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Diebold-Li Yield Curve Forecasting App", layout="wide")

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
    name = filename.upper()
    stem = re.sub(r"\.(CSV|XLSX|XLS)$", "", name)

    if stem in MATURITY_MAP:
        return stem

    for code in ORDERED_CODES:
        pattern = rf"(^|[^A-Z0-9]){code}([^A-Z0-9]|$)"
        if re.search(pattern, name):
            return code

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
        yield_col = None
        for c in df.columns:
            if c != date_col:
                numeric = pd.to_numeric(df[c], errors="coerce")
                if numeric.notna().sum() > 0:
                    yield_col = c
                    break

    if yield_col is None:
        raise ValueError(f"Could not identify yield column in {filename}")

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
            raise ValueError(f"Duplicate series detected: {code}. Upload one file per maturity.")
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


def fit_factors_fixed_lambda(yields, maturities, lambd):
    y = np.asarray(yields, dtype=float)
    X = ns_loadings(maturities, lambd)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ betas
    errors = y - fitted
    sse = float(np.sum(errors ** 2))
    return betas, fitted, errors, sse, X


def estimate_factors_fixed_lambda(model_data, yield_cols, lambd):
    maturities = [MATURITY_MAP[c] for c in yield_cols]
    rows = []

    progress = st.progress(0)
    total = len(model_data)

    for i, (_, row) in enumerate(model_data.iterrows()):
        yields = row[yield_cols].to_numpy(dtype=float)
        betas, fitted, errors, sse, _ = fit_factors_fixed_lambda(yields, maturities, lambd)

        rows.append({
            "Date": row["Date"],
            "beta0_level": float(betas[0]),
            "beta1_slope": float(betas[1]),
            "beta2_curvature": float(betas[2]),
            "SSE": sse,
        })
        progress.progress((i + 1) / total)

    return pd.DataFrame(rows)


def create_forecast_dataset(factors, horizon):
    df = factors.copy()
    for col in ["beta0_level", "beta1_slope", "beta2_curvature"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_future"] = df[col].shift(-horizon)
    return df.dropna().reset_index(drop=True)


def evaluate_factor_forecasts(forecast_df):
    results = []
    factor_cols = ["beta0_level", "beta1_slope", "beta2_curvature"]

    split = int(len(forecast_df) * 0.8)
    train = forecast_df.iloc[:split]
    test = forecast_df.iloc[split:]

    for factor in factor_cols:
        target = f"{factor}_future"

        # Random walk: future factor forecast equals current factor
        rw_pred = test[factor].values
        actual = test[target].values

        results.append({
            "Target": target,
            "Model": "Random Walk",
            "RMSE": np.sqrt(mean_squared_error(actual, rw_pred)),
            "MAE": mean_absolute_error(actual, rw_pred),
        })

        # Linear regression using lagged/current factor predictors
        X_cols = [
            "beta0_level", "beta1_slope", "beta2_curvature",
            "beta0_level_lag1", "beta1_slope_lag1", "beta2_curvature_lag1"
        ]

        model = LinearRegression()
        model.fit(train[X_cols], train[target])
        pred = model.predict(test[X_cols])

        results.append({
            "Target": target,
            "Model": "Linear Regression",
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAE": mean_absolute_error(actual, pred),
        })

    return pd.DataFrame(results)


st.title("Diebold-Li Yield Curve Forecasting App")

st.markdown(
    """
This version uses the **research-standard Diebold-Li approach**:

1. Fix λ instead of estimating a new unstable λ every date.
2. Estimate Nelson-Siegel factors β0, β1, and β2 for each month.
3. Use those factor time series for forecasting.

The model is:

$$
y(\\tau)=\\beta_0+\\beta_1\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau}\\right)
+\\beta_2\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau}-e^{-\\lambda\\tau}\\right)
$$
"""
)

with st.sidebar:
    st.header("Settings")
    frequency = st.radio("Use data frequency", ["Monthly last observation", "Daily"], index=0)
    lambda_value = st.number_input(
        "Fixed λ",
        min_value=0.001,
        value=0.0609,
        step=0.001,
        format="%.4f",
        help="Diebold-Li commonly fixes lambda. 0.0609 is a standard monthly value."
    )
    forecast_horizon = st.number_input("Forecast horizon, rows/months", min_value=1, value=1, step=1)

st.info(
    "Upload FRED files such as DGS3MO.xlsx, DGS6MO.xlsx, DGS1.xlsx, DGS2.xlsx, DGS10.xlsx, DGS30.xlsx. "
    "The app reads the FRED `Daily` sheet automatically."
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
st.dataframe(file_info, use_container_width=True)

mapping_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity in years": [MATURITY_MAP[c] for c in yield_cols],
})
st.dataframe(mapping_df, use_container_width=True)

if len(yield_cols) < 4:
    st.warning("Use at least 4 maturities. Six or more is better.")

st.header("2. Merged yield data")
st.write(f"Rows after merge: {len(merged)}")
st.dataframe(merged.head(25), use_container_width=True)

model_data = monthly_last_observation(merged) if frequency == "Monthly last observation" else merged.copy()

st.header("3. Data used for estimation")
st.write(f"Frequency selected: **{frequency}**")
st.write(f"Rows used: {len(model_data)}")
st.dataframe(model_data.head(25), use_container_width=True)

st.header("4. Single-date fixed-λ Nelson-Siegel calculation")

date_options = model_data["Date"].dt.strftime("%Y-%m-%d").tolist()
selected_date = st.selectbox("Choose date", date_options, index=len(date_options) - 1)

row = model_data[model_data["Date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
maturities = [MATURITY_MAP[c] for c in yield_cols]
actual_yields = row[yield_cols].to_numpy(dtype=float)

betas, fitted, errors, sse, X = fit_factors_fixed_lambda(actual_yields, maturities, lambda_value)

cols = st.columns(5)
cols[0].metric("Fixed λ", f"{lambda_value:.4f}")
cols[1].metric("β0 Level", f"{betas[0]:.4f}")
cols[2].metric("β1 Slope", f"{betas[1]:.4f}")
cols[3].metric("β2 Curvature", f"{betas[2]:.4f}")
cols[4].metric("SSE", f"{sse:.6f}")

st.subheader("4A. Actual yields")
actual_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity τ": maturities,
    "Actual yield": actual_yields,
})
st.dataframe(actual_df, use_container_width=True)

st.subheader("4B. Nelson-Siegel loading table")
loading_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity τ": maturities,
    "X0 β0 loading": X[:, 0],
    "X1 β1 loading": X[:, 1],
    "X2 β2 loading": X[:, 2],
})
st.dataframe(loading_df, use_container_width=True)

st.subheader("4C. Fitted values and errors")
fit_df = pd.DataFrame({
    "FRED code": yield_cols,
    "Maturity τ": maturities,
    "Actual yield": actual_yields,
    "Fitted yield": fitted,
    "Error actual - fitted": errors,
    "Squared error": errors ** 2,
})
st.dataframe(fit_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(maturities, actual_yields, marker="o", label="Actual")
ax.plot(maturities, fitted, marker="o", label="Diebold-Li fitted")
ax.set_xlabel("Maturity in years")
ax.set_ylabel("Yield, percent")
ax.set_title(f"Fixed-λ Nelson-Siegel fit: {selected_date}")
ax.legend()
st.pyplot(fig)

st.markdown(
    """
**What is happening here?**

Because λ is fixed, the Nelson-Siegel model becomes a normal regression:

$$
Yield = \\beta_0 X_0 + \\beta_1 X_1 + \\beta_2 X_2
$$

So the app estimates only β0, β1, and β2 for each date.  
This avoids the unstable huge-beta problem from estimating λ every month.
"""
)

st.header("5. Estimate Diebold-Li factors for every date")

if st.button("Run fixed-λ factor estimation"):
    with st.spinner("Estimating factors for all dates..."):
        factors = estimate_factors_fixed_lambda(model_data, yield_cols, lambda_value)
        st.session_state["factors"] = factors

if "factors" in st.session_state:
    factors = st.session_state["factors"]

    st.subheader("Estimated factor table")
    st.dataframe(factors, use_container_width=True)

    st.subheader("Factor chart")
    st.line_chart(factors.set_index("Date")[["beta0_level", "beta1_slope", "beta2_curvature"]])

    st.download_button(
        "Download Diebold-Li factors CSV",
        data=factors.to_csv(index=False).encode("utf-8"),
        file_name="diebold_li_factors.csv",
        mime="text/csv",
    )

    st.header("6. Forecasting dataset")

    forecast_df = create_forecast_dataset(factors, forecast_horizon)
    st.write("This creates future factor targets and lagged predictors for forecasting.")
    st.dataframe(forecast_df, use_container_width=True)

    st.download_button(
        "Download forecasting dataset CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_dataset.csv",
        mime="text/csv",
    )

    if len(forecast_df) >= 20:
        st.header("7. Simple benchmark forecast comparison")
        st.write("This compares Random Walk vs Linear Regression for factor forecasting.")
        metrics = evaluate_factor_forecasts(forecast_df)
        st.dataframe(metrics, use_container_width=True)

        st.download_button(
            "Download forecast metrics CSV",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="forecast_metrics.csv",
            mime="text/csv",
        )
    else:
        st.warning("Not enough rows for benchmark forecasting comparison.")
