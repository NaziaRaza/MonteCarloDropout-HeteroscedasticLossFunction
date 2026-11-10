import shap
import torch
import matplotlib.pyplot as plt

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

# Complete prediction function for computing mean, epistemic and aleatoric variances
def predict_function(model, input_data, mc_samples=50):
    model.eval()
    mean_list, log_var_list = [], []
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_data).to(dtype=torch.float32).to(device)
        for _ in range(mc_samples):
            output = model(input_tensor)
            mean, log_var = output
            mean_list.append(mean)
            log_var_list.append(log_var)    

    mean_stack    = torch.stack(mean_list)                # [mc_samples, batch, output]
    mean_pred     = mean_stack.mean(dim=0).cpu().numpy()  # [batch, output]
    epistemic_var = mean_stack.var(dim=0).cpu().numpy()   # [batch, output]

    log_var_stack = torch.stack(log_var_list)             # [mc_samples, batch, output]
    aleatoric_var = torch.exp(log_var_stack).mean(dim=0).cpu().numpy()

    return mean_pred, epistemic_var, aleatoric_var

# Compute SHAP values for one output variables
def get_shap_values(model, X_background, X_sample, output_idx, plot):

    def predict_fn_mean(x, output_idx):
        mean_pred, _, _ = predict_function(model, x)
        return mean_pred[:, output_idx]

    def predict_fn_epistemic(x, output_idx):
        _, epistemic_var, _ = predict_function(model, x)
        return epistemic_var[:, output_idx]

    def predict_fn_aleatoric(x, output_idx):
        _, _, aleatoric_var = predict_function(model, x)
        return aleatoric_var[:, output_idx]

    explainer_mean = shap.KernelExplainer(
        model=lambda x: predict_fn_mean(x, output_idx),
        data=X_background,
        link="identity"
    )

    explainer_epi = shap.KernelExplainer(
        model=lambda x: predict_fn_epistemic(x, output_idx),
        data=X_background,
        link="identity"
    )

    explainer_ale = shap.KernelExplainer(
        model=lambda x: predict_fn_aleatoric(x, output_idx),
        data=X_background,
        link="identity"
    )
    shap_values_epi  = explainer_epi.shap_values(X_sample, nsamples=100)
    shap_values_mean = explainer_mean.shap_values(X_sample, nsamples=100)
    shap_values_ale  = explainer_ale.shap_values(X_sample, nsamples=100)

    if plot:
        feature_names = ['$V_{bus}$', '$I_{sga}$', '$I_{sgb}$', '$I_{bata}$', '$I_{batb}$',
            '$I_{sca}$', '$I_{scb}$', '$V_{sca}$', '$V_{scb}$', '$e$', '$P_{ppl}$', '$P_{cpl}$', '$V_{ref}$' ]

        fig, axes = plt.subplots(
            1, 3, figsize=(10, 3.5),
            gridspec_kw={'width_ratios': [1.2, 1.2, 2.0]}  # ← make heatmap 2× wider
        )
        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(hspace=0.1)

        # Beeswarm plot for mean prediction
        plt.sca(axes[0])
        shap.summary_plot(shap_values_mean, X_sample, feature_names=feature_names, plot_type="violin", plot_size=None, show=False)
        axes[0].set_title("Mean Prediction", fontsize=13)
        axes[0].set_xlabel("SHAP value (impact on model output)", fontsize=12)

        # Bar plot for epistemic uncertainty
        plt.sca(axes[1])
        shap.summary_plot(shap_values_epi, X_sample, feature_names=feature_names, plot_type="bar", plot_size=None, show=False)
        axes[1].set_title("Epistemic Uncertainty", fontsize=13)
        axes[1].set_xlabel("$\mathrm{mean}(|\mathrm{SHAP\ value}|)$", fontsize=12)

        # Wrap your SHAP results and data
        shap_exp_ale = shap.Explanation(
            values=shap_values_ale,
            data=X_sample,
            feature_names=feature_names
        )
        plt.sca(axes[2])
        shap.plots.heatmap(shap_exp_ale, ax=axes[2], max_display=15, instance_order=shap_exp_ale.sum(1), show=False)
        axes[2].set_title("Aleatoric Uncertainty", fontsize=13)
        axes[2].set_xlabel("Instances", fontsize=12)

        # Remove SHAP heatmap colorbar title
        for ax in fig.axes:
            for cax in ax.figure.get_axes():
                if hasattr(cax, "get_ylabel") and "SHAP value" in cax.get_ylabel():
                    cax.set_ylabel("")  # remove the label text

        plt.tight_layout()
        plt.show()

        # plt.savefig(rf".\IEEE Revision Codes\Journal Plots\shap_{output_idx}.pdf", bbox_inches = 'tight', dpi=300)
        

    return shap_values_mean, shap_values_epi, shap_values_ale