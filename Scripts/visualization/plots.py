import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib import gridspec

def residual_plot_with_metrics(fig, position, y_pred, y_true, label):
    color_picked = "#5D7251"  # simplified hex
    residuals = y_true - y_pred
    r2              = r2_score(y_true, y_pred)
    nrmse           = np.sqrt(((y_true - y_pred)**2).mean(axis=0)) / np.abs(y_true).mean(axis=0)
    weighted_mape   = (np.abs(y_true - y_pred).sum(axis=0) / np.abs(y_true).sum(axis=0))

    gs = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=position,
        width_ratios=[4, 0.5, 0.1],
        height_ratios=[0.5, 4, 0.1]
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_xkde = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_ykde = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # scatter (subsample for readability)
    frac = 0.3
    n = len(y_pred)
    idx = np.random.choice(n, size=int(frac * n), replace=False)

    ax_main.scatter(y_pred[idx], residuals[idx], color=color_picked, s=10, alpha=0.6)
    ax_main.axhline(0, ls='--', color='gray', lw=1)
    ax_main.set_xlabel(fr"$\hat{{{label}}}$", fontsize=10)
    ax_main.set_ylabel(fr"${label} - \hat{{{label}}}$", fontsize=10)
    ax_main.grid(True, linestyle='--', alpha=0.1)

    # metrics text
    metrics_text = (
        f"$R^2$ = {r2:.3f}\n"
        f"NMSE = {nrmse:.5f}\n"
        f"WMAPE = {weighted_mape:.5f}"
    )
    ax_main.text(
        0.1, 0.8, metrics_text,
        transform=ax_main.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
    )

    # Replace hist with KDE plots
    sns.kdeplot(y_pred, ax=ax_xkde, fill=True, color='y', alpha=0.2, linewidth=1)
    sns.kdeplot(y=residuals, ax=ax_ykde, fill=True, color='y', alpha=0.2, linewidth=1)

    # --- X KDE (top) ---
    ax_xkde.tick_params(
        axis='x', which='both',
        bottom=False, labelbottom=False
    )
    # ax_xkde.tick_params(
    #     axis='y', which='both',
    #     left=False, labelleft=False
    # )

    # --- Y KDE (right) ---
    ax_ykde.tick_params(
        axis='y', which='both',
        left=False, labelleft=False
    )
    ax_ykde.tick_params(
        axis='x', which='both',
        bottom=False, labelbottom=False
    )

    # Remove spines only for KDE plots
    for ax in [ax_xkde, ax_ykde]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    return r2, nrmse, weighted_mape