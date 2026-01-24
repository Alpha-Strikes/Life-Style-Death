import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

#add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_URL

from data_prep import load_raw_data, clean_dataset





def basic_eda(data_url: str = DATASET_URL) -> None:
    if not os.path.exists("figures"):
        os.makedirs("figures")
    df = clean_dataset(load_raw_data(data_url))

    #dist of age_at_death
    plt.figure(figsize=(6, 4))
    sns.histplot(df["age_at_death"], kde=True)
    plt.title("Distribution of Age at Death")
    plt.xlabel("Age at death")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "age_at_death_distribution.png"))
    plt.close()

    #####boxplots
    if "gender" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="gender", y="age_at_death", data=df)
        plt.title("Age at Death by Gender")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "age_by_gender.png"))
        plt.close()
    if "occupation_type" in df.columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x="occupation_type", y="age_at_death", data=df)
        plt.xticks(rotation=45, ha="right")
        plt.title("Age at Death by Occupation Type")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "age_by_occupation.png"))
        plt.close()

    #####pairplots
    numeric_cols = ["avg_work_hours_per_day","avg_rest_hours_per_day","avg_sleep_hours_per_day","avg_exercise_hours_per_day","age_at_death"]
    present_numeric = [c for c in numeric_cols if c in df.columns]
    if len(present_numeric) > 1:
        sns.pairplot(df[present_numeric])
        plt.savefig(os.path.join("figures", "pairplot_lifestyle_age.png"))
        plt.close()


if __name__ == "__main__":
    basic_eda()

