import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load run-by-run summary data
    df = pd.read_csv("simulation_summary.csv")

    # ---------------------------------------------------------------------
    # 2. Choose which columns are "settings" and which are "results"
    # ---------------------------------------------------------------------
    # We'll group by just WAIT_SHARING for this example. Adjust as desired.
    groupby_settings = ["WAIT_SHARING", "FOOD_SHARING"]

    # Columns that hold "results" we want to analyze:
    result_cols = ["PeakPopA", "PeakPopB", "TotalPopA", "TotalPopB"]

    # ---------------------------------------------------------------------
    # 3. Group by the chosen settings
    # ---------------------------------------------------------------------
    grouped = df.groupby(groupby_settings)

    # Create a summary DataFrame with mean, std, min, max, count
    group_summary = grouped[result_cols].agg(["mean", "std", "min", "max", "count"])
    print("\n=== Grouped Summary of Results by Settings ===")
    print(group_summary)

    # ---------------------------------------------------------------------
    # 4. Generate some plots per group
    # ---------------------------------------------------------------------
    for settings_values, subdf in grouped:
        # `settings_values` is a scalar or tuple of values if multiple groupby columns
        # `subdf` is the subset of df that matches those setting values

        # Convert group settings into a neat string for printing and plot titles
        # e.g. WAIT_SHARING=200
        if isinstance(settings_values, tuple):
            # if we had more than one grouping column, they'd appear here
            settings_str = ", ".join(f"{name}={val}"
                                     for name, val in zip(groupby_settings, settings_values))
        else:
            # if there's only one grouping column, settings_values won't be a tuple
            settings_str = f"{groupby_settings[0]}={settings_values}"

        print(f"\n--- Group: {settings_str} ---")
        # Print descriptive stats for our result columns in this group
        print(subdf[result_cols].describe())

        # 4A. Box Plot: PeakPopA vs. PeakPopB
        plt.figure()
        subdf[["PeakPopA", "PeakPopB"]].boxplot()
        plt.title(f"Box Plot: PeakPopA vs. PeakPopB\n({settings_str})")
        plt.ylabel("Peak Population")
        plt.grid(True)
        plt.show()

        # 4B. Box Plot: TotalPopA vs. TotalPopB
        plt.figure()
        subdf[["TotalPopA", "TotalPopB"]].boxplot()
        plt.title(f"Box Plot: TotalPopA vs. TotalPopB\n({settings_str})")
        plt.ylabel("Total Population (Sum Over Snapshots)")
        plt.grid(True)
        plt.show()

        # 4C. Survival Ratio (Wins Ratio) for each group
        # We assume your summary has 'AWin'=1 if A survived, else 0; same for 'BWin'.
        # Let's compute how many runs in this group had A survive, and B survive.
        a_win_count = subdf["AWin"].sum()
        b_win_count = subdf["BWin"].sum()
        group_size = len(subdf)

        # We define "survival ratio" as (# of runs survived by group) / (total runs in group).
        a_survival_ratio = a_win_count / group_size
        b_survival_ratio = b_win_count / group_size

        print(f"In group {settings_str}, Group A survived in {a_win_count} of {group_size} runs.")
        print(f"In group {settings_str}, Group B survived in {b_win_count} of {group_size} runs.")

        plt.figure()
        plt.bar(["A Survival Ratio", "B Survival Ratio"],
                [a_survival_ratio, b_survival_ratio])
        plt.ylim(0, 1)  # Ratios will be between 0 and 1
        plt.title(f"Survival Ratio for Each Group\n({settings_str})")
        plt.ylabel("Fraction of Runs Survived")
        plt.grid(True)
        plt.show()

    print("\nDone grouping by simulation settings.")

if __name__ == "__main__":
    main()
