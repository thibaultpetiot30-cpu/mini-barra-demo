import streamlit as st
import pandas as pd
import numpy as np

st.title("Mini Barra Risk Model Demo 📊")

st.write("""
Upload a CSV file with the following columns:
- **date** (YYYY-MM-DD)
- **factor_Mkt** (market factor return)
- **factor_SMB** (size factor return)
- **asset** (asset name)
- **return** (asset return)
""")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data preview", df.head())

    # Pivot to get asset returns matrix
    pivot = df.pivot(index="date", columns="asset", values="return").dropna()
    factors = df.groupby("date")[["factor_Mkt","factor_SMB"]].mean().loc[pivot.index]

    # Estimate betas (simple regression)
    X = np.column_stack([factors["factor_Mkt"], factors["factor_SMB"]])
    betas = {}
    spec_vars = {}
    for asset in pivot.columns:
        y = pivot[asset].values
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        betas[asset] = b
        resid = y - X @ b
        spec_vars[asset] = resid.var()

    betas_df = pd.DataFrame(betas, index=["Mkt","SMB"]).T
    st.subheader("Estimated Exposures (Betas)")
    st.dataframe(betas_df.round(3))

    # Factor covariance
    F_cov = np.cov(X.T)
    st.subheader("Factor Covariance")
    st.write(pd.DataFrame(F_cov, index=["Mkt","SMB"], columns=["Mkt","SMB"]).round(4))

    # Portfolio weights (equal-weight example)
    w = np.ones(len(pivot.columns)) / len(pivot.columns)
    X_mat = betas_df.values
    Delta = np.diag(list(spec_vars.values()))
    cov_assets = X_mat @ F_cov @ X_mat.T + Delta
    var_p = w.T @ cov_assets @ w
    vol_p = np.sqrt(var_p)

    st.subheader("Portfolio Risk")
    st.write(f"Variance: {var_p:.4f}")
    st.write(f"Volatility: {vol_p:.2%}")

    # Pie chart of risk contributions (simplified)
    import matplotlib.pyplot as plt
    factor_var = w.T @ (X_mat @ F_cov @ X_mat.T) @ w
    spec_var = w.T @ Delta @ w
    labels = ["Factor risk", "Specific risk"]
    values = [factor_var, spec_var]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Risk Decomposition")
    st.pyplot(fig)
