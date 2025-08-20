# chart.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Synthetic data (reproducible) ----------
rng = np.random.default_rng(42)

channels = ["Email", "Chat", "Phone", "Social"]

# Log-normal response times (minutes), heavier tail for Email/Social
def make_times(mu, sigma, n):
    # mu = mean of log, sigma = std of log; median ≈ exp(mu)
    return rng.lognormal(mean=mu, sigma=sigma, size=n)

n_per = 500
data = pd.DataFrame({
    "channel": np.repeat(channels, n_per),
    "response_minutes": np.concatenate([
        make_times(mu=3.4, sigma=0.6, n=n_per),  # Email  ~ median ~ 30 min
        make_times(mu=2.5, sigma=0.4, n=n_per),  # Chat   ~ median ~ 12 min
        make_times(mu=2.8, sigma=0.5, n=n_per),  # Phone  ~ median ~ 16 min
        make_times(mu=3.1, sigma=0.7, n=n_per),  # Social ~ median ~ 22 min
    ])
})

# Optional cap for extreme outliers to keep the plot readable
data["response_minutes"] = data["response_minutes"].clip(0, 120)

# ---------- Seaborn styling ----------
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0)

# 512x512 px at 64 dpi -> figsize = (8, 8) inches
plt.figure(figsize=(8, 8))

# ---------- Violin plot ----------
ax = sns.violinplot(
    data=data,
    x="channel",
    y="response_minutes",
    palette="Set2",
    inner="quartile",     # show median & quartiles
    cut=0,                # do not extend past data min/max
    linewidth=1
)

# Titles and labels (presentation-ready)
ax.set_title("Customer Support Response Time by Channel", pad=14, weight="bold")
ax.set_xlabel("Support Channel")
ax.set_ylabel("Response Time (minutes)")
ax.set_ylim(0, 120)

# Footnote to clarify synthetic data
ax.text(
    0.5, -0.12,
    "Synthetic data for demonstration (n = {}).".format(len(data)),
    transform=ax.transAxes, ha="center", va="top", fontsize=10
)

# ---------- Export (exact spec) ----------
# NOTE: Using dpi=64 with 8x8 in -> 512x512 px. The assignment asks for bbox_inches='tight'.
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()
