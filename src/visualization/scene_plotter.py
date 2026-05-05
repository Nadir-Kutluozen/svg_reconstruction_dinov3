import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR
from src.probing.scene_experiments import run_forward_probing, run_reverse_probing, USED_DIMS

# Style constants
PLOT_SIZE = 0.55
PERSON_COLOR     = "#1f4e9b"
PERSON_RAND_COL  = "#a8b8d4"
ANIMAL_COLOR     = "#d97a1a"
ANIMAL_RAND_COL  = "#f0c89e"
PRE_COLOR        = "#1f4e9b"
RAND_COLOR       = "#d97a1a"

def make_forward_figs(feature_mode="cls"):
    print(f"Generating Forward Decoding Figures using {feature_mode}...")
    
    # We aggregate over the two domains to get within-domain summary
    rho_pre_island, _ = run_forward_probing("island", "island", "pre", mode=feature_mode)
    rho_pre_western, _ = run_forward_probing("western", "western", "pre", mode=feature_mode)
    rho_rand_island, _ = run_forward_probing("island", "island", "rand", mode=feature_mode)
    rho_rand_western, _ = run_forward_probing("western", "western", "rand", mode=feature_mode)
    
    # We only care about the USED dims, which is indices 0-15 and 28-31
    # Actually wait, run_forward_probing returns (r2, rho), so I need [1]
    _, rho_pre_island = run_forward_probing("island", "island", "pre", mode=feature_mode)
    _, rho_pre_western = run_forward_probing("western", "western", "pre", mode=feature_mode)
    _, rho_rand_island = run_forward_probing("island", "island", "rand", mode=feature_mode)
    _, rho_rand_western = run_forward_probing("western", "western", "rand", mode=feature_mode)

    rho_pre = 0.5 * (rho_pre_island[USED_DIMS] + rho_pre_western[USED_DIMS])
    rho_rand = 0.5 * (rho_rand_island[USED_DIMS] + rho_rand_western[USED_DIMS])

    # Labels for the 20 dims
    labels = [
        "p0_x", "p0_z", "p0_facing", "p0_skin_h",
        "p1_x", "p1_z", "p1_facing", "p1_skin_h",
        "p2_x", "p2_z", "p2_facing", "p2_skin_h",
        "p3_x", "p3_z", "p3_facing", "p3_skin_h",
        "dolphin_x", "dolphin_z", "turtle_x", "turtle_z"
    ]
    groups = ["person"] * 16 + ["animal"] * 4

    pre_color  = [PERSON_COLOR    if g == "person" else ANIMAL_COLOR    for g in groups]
    rand_color = [PERSON_RAND_COL if g == "person" else ANIMAL_RAND_COL for g in groups]

    fig, ax = plt.subplots(figsize=PLOT_SIZE * np.array((14, 5.5)))

    x = np.arange(len(labels))
    bar_w = 0.4

    ax.bar(x - bar_w/2, rho_pre,  width=bar_w, color=pre_color, edgecolor="white", linewidth=0.6)
    ax.bar(x + bar_w/2, rho_rand, width=bar_w, color=rand_color, edgecolor="white", linewidth=0.6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Pearson $\rho$")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PERSON_COLOR,    label="person, pretrained"),
        Patch(facecolor=PERSON_RAND_COL, label="person, random"),
        Patch(facecolor=ANIMAL_COLOR,    label="animal, pretrained"),
        Patch(facecolor=ANIMAL_RAND_COL, label="animal, random"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8, ncol=4)
    plt.title(f"Forward Probing: Pearson Correlation (Feature: {feature_mode})", fontsize=12)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"fig_forward_decoding_{feature_mode}.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Saved Forward Decoding Plot to {out_path}")

def make_reverse_fig():
    print("Generating Reverse Probing Whitening Figure... (This runs a sweep and will take ~30 seconds)")
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=PLOT_SIZE * np.array((12, 4.5)),
        gridspec_kw=dict(wspace=0.28),
        sharey=True,
    )

    sweep_ks = [32, 64, 128, 256, 768]
    
    for ax, mode, title in [
        (axes[0], "cls",  "CLS token"),
        (axes[1], "patches_mean",  "Mean-pooled patches"),
    ]:
        print(f"Sweeping {mode} features...")
        pre_raw_island = run_reverse_probing("island", "pre", mode=mode)["raw"]
        pre_raw_western = run_reverse_probing("western", "pre", mode=mode)["raw"]
        rand_raw_island = run_reverse_probing("island", "rand", mode=mode)["raw"]
        rand_raw_western = run_reverse_probing("western", "rand", mode=mode)["raw"]
        
        raw_pre = 0.5 * (pre_raw_island + pre_raw_western)
        raw_rand = 0.5 * (rand_raw_island + rand_raw_western)
        
        pre_vals = []
        rand_vals = []
        for k in sweep_ks:
            pw_pre_i = run_reverse_probing("island", "pre", mode=mode, n_pca_whiten=k)["pca_whitened"]
            pw_pre_w = run_reverse_probing("western", "pre", mode=mode, n_pca_whiten=k)["pca_whitened"]
            pre_vals.append(0.5 * (pw_pre_i + pw_pre_w))
            
            pw_ran_i = run_reverse_probing("island", "rand", mode=mode, n_pca_whiten=k)["pca_whitened"]
            pw_ran_w = run_reverse_probing("western", "rand", mode=mode, n_pca_whiten=k)["pca_whitened"]
            rand_vals.append(0.5 * (pw_ran_i + pw_ran_w))

        ax.plot(sweep_ks, pre_vals,  marker="o", lw=2.0, color=PRE_COLOR, label="pretrained")
        ax.plot(sweep_ks, rand_vals, marker="s", lw=2.0, color=RAND_COLOR, label="random init")

        ax.axhline(raw_pre,  color=PRE_COLOR,  ls="--", lw=1.0, alpha=0.6)
        ax.axhline(raw_rand, color=RAND_COLOR, ls="--", lw=1.0, alpha=0.6)
        
        ax.text(750, raw_pre + 0.005, f"raw: {raw_pre:.2f}", color=PRE_COLOR, fontsize=8, ha="right", va="bottom")
        ax.text(750, raw_rand + 0.005, f"raw: {raw_rand:.2f}", color=RAND_COLOR, fontsize=8, ha="right", va="bottom")

        ax.set_xscale("log")
        ax.set_xticks([32, 64, 128, 256, 768])
        ax.set_xticklabels(["32", "64", "128", "256", "768"])
        ax.set_xlabel(r"PCA-whitening cutoff $k$")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(min(min(rand_vals), raw_rand) - 0.02, max(max(pre_vals), raw_pre) + 0.05)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel(r"$R^2_\mathrm{rev}$")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fig_reverse_whitening.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Saved Reverse Whitening Plot to {out_path}")
