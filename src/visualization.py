import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import ANALYSIS_DATA_DIR


def require_csv(filename: str) -> str:
    path = os.path.join(ANALYSIS_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {filename}. Run src/feature_engineering.py to generate analysis outputs first."
        )
    return path


def generate_static_plots() -> None:
    national = pd.read_csv(require_csv("national_monthly_summary.csv"))
    state_master = pd.read_csv(require_csv("state_master_full.csv"))
    pareto = pd.read_csv(require_csv("pareto_activity.csv"))

    print("Generating static visualizations...")

    plt.figure(figsize=(11, 5))
    plt.plot(
        national["year_month"], national["E_total"], label="Enrollments", linewidth=2.5
    )
    plt.plot(
        national["year_month"],
        national["D_total"],
        label="Demographic Updates",
        linewidth=2,
    )
    plt.plot(
        national["year_month"],
        national["B_total"],
        label="Biometric Updates",
        linewidth=2,
    )
    plt.xticks(rotation=45)
    plt.title("National UIDAI Activity Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(ANALYSIS_DATA_DIR, "fig_national_activity_trend.png"), dpi=200
    )
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=state_master,
        x="AMI",
        y="UPI",
        hue="Policy_Category",
        size="VSI",
        sizes=(60, 350),
        alpha=0.85,
    )
    plt.axvline(state_master["AMI"].median(), linestyle="--", color="gray")
    plt.axhline(state_master["UPI"].median(), linestyle="--", color="gray")
    plt.title("Governance Quadrant")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DATA_DIR, "fig_governance_quadrant.png"), dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(pareto["activity_rank"], pareto["Total_Activity"], color="#102a43")
    ax1.set_xlabel("State Rank by Activity")
    ax1.set_ylabel("Total Activity")
    ax2 = ax1.twinx()
    ax2.plot(pareto["activity_rank"], pareto["cum_share"], color="#ea580c", linewidth=3)
    ax2.set_ylabel("Cumulative Share")
    plt.title("Pareto Analysis of UIDAI Activity Concentration")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DATA_DIR, "fig_pareto_activity.png"), dpi=200)
    plt.close()

    print(f"Static visualizations saved to {ANALYSIS_DATA_DIR}")


if __name__ == "__main__":
    generate_static_plots()
