import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def build_interaction_design(X: np.ndarray, idx: list[int]) -> tuple[np.ndarray, list[str]]:
    pairs = list(combinations(idx, 2))
    X_design = X.copy()
    for i, j in pairs:
        X_design = np.column_stack((X_design, X[:, i] * X[:, j]))
    names = [f"x{i}" for i in range(X.shape[1])] + [f"x{i}*x{j}" for i, j in pairs]
    return X_design, names


def save_lassocv_curve(lasso_cv: LassoCV, out_path: str) -> None:
    cv_mean = lasso_cv.mse_path_.mean(axis=1)
    cv_std = lasso_cv.mse_path_.std(axis=1)
    alphas = lasso_cv.alphas_

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(alphas, cv_mean, color="#1f77b4", linewidth=2, label="Mean CV MSE")
    ax.fill_between(
        alphas,
        cv_mean - cv_std,
        cv_mean + cv_std,
        color="#1f77b4",
        alpha=0.2,
        label=r"$\pm$1 std",
    )
    ax.axvline(
        lasso_cv.alpha_,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Selected alpha = {lasso_cv.alpha_:.4g}",
    )
    ax.set_title("LassoCV Selection Curve (Task 7)")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("CV Mean Squared Error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_true_vs_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=24, alpha=0.75, edgecolor="none", color="#2ca02c")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=2, label="Ideal: y_pred = y_true")
    ax.set_title("Best Model: Interaction Lasso\nTrue vs Predicted")
    ax.set_xlabel("True quality")
    ax.set_ylabel("Predicted quality")
    ax.text(
        0.04,
        0.96,
        f"Residual mean = {residuals.mean():.3f}\nResidual std = {residuals.std():.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    train = np.loadtxt("wine_training.csv", delimiter=",")
    test = np.loadtxt("wine_test.csv", delimiter=",")

    X_train, y_train = train[:, :11], train[:, 11]
    X_test, y_test = test[:, :11], test[:, 11]

    # Use training statistics to standardize both train and test.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Part 2: First-order OLS
    ols_first = LinearRegression().fit(X_train_s, y_train_s)
    mse_first = mean_squared_error(y_test_s, ols_first.predict(X_test_s))

    # Part 3: Top-5 absolute first-order coefficients
    top5_idx = np.argsort(np.abs(ols_first.coef_))[-5:][::-1]

    # Part 4: Second-order (quadratic, no interactions) OLS
    X_train_quad = np.hstack((X_train_s, X_train_s**2))
    X_test_quad = np.hstack((X_test_s, X_test_s**2))
    ols_quad = LinearRegression().fit(X_train_quad, y_train_s)
    mse_quad = mean_squared_error(y_test_s, ols_quad.predict(X_test_quad))

    # Part 6: First-order + selected interactions OLS
    interaction_idx = [1, 4, 6, 9, 10]
    X_train_int, int_names = build_interaction_design(X_train_s, interaction_idx)
    X_test_int, _ = build_interaction_design(X_test_s, interaction_idx)
    ols_int = LinearRegression().fit(X_train_int, y_train_s)
    mse_int = mean_squared_error(y_test_s, ols_int.predict(X_test_int))

    # Part 7: Lasso model selection on interaction model
    alphas = np.logspace(-4, 2, 300)
    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=200000, random_state=0).fit(
        X_train_int, y_train_s
    )
    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha, max_iter=200000).fit(X_train_int, y_train_s)
    y_pred_test_s = lasso.predict(X_test_int)
    mse_lasso = mean_squared_error(y_test_s, y_pred_test_s)
    nonzero = [
        (name, coef) for name, coef in zip(int_names, lasso.coef_) if abs(coef) > 1e-10
    ]

    model_mses = {
        "first_order_ols": mse_first,
        "quadratic_ols": mse_quad,
        "interaction_ols": mse_int,
        "interaction_lasso": mse_lasso,
    }
    best_model = min(model_mses, key=model_mses.get)

    # Save figures for report.
    save_lassocv_curve(lasso_cv, "fig_lassocv_selection_curve.png")
    y_pred_test = y_scaler.inverse_transform(y_pred_test_s.reshape(-1, 1)).ravel()
    save_true_vs_pred_scatter(y_test, y_pred_test, "fig_interaction_lasso_true_vs_pred.png")

    print("Part 1: transform test set with TRAINING means/stds only.")
    print(f"Part 2 test MSE (first-order OLS): {mse_first:.12f}")
    print(f"Part 3 top-5 coefficient indices: {top5_idx.tolist()}")
    print(f"Part 4 test MSE (quadratic OLS): {mse_quad:.12f}")
    print(f"Part 6 test MSE (interaction OLS): {mse_int:.12f}")
    print(f"Part 7 selected alpha (LassoCV): {best_alpha:.14f}")
    print(f"Part 7 test MSE (interaction Lasso): {mse_lasso:.12f}")
    print("Part 7 nonzero coefficients:")
    for name, coef in nonzero:
        print(f"  {name}: {coef:.12f}")
    print(f"Part 8 best model by test MSE: {best_model} ({model_mses[best_model]:.12f})")
    print("Saved figure: fig_lassocv_selection_curve.png")
    print("Saved figure: fig_interaction_lasso_true_vs_pred.png")


if __name__ == "__main__":
    main()
