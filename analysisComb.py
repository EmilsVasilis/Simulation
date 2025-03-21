import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load run-by-run summary data
    df = pd.read_csv("simulation_summary.csv")

    # We only care about WAIT_SHARING in [0, 100, 200]
    # (If your dataset has these exact values. Otherwise adjust as needed.)
    relevant_waits = [0, 100, 200]

    # We'll assume FOOD_SHARING in [1, 2, 3].
    relevant_food_sharing = [1, 2, 3]

    # 2. Loop over each WAIT_SHARING value and create bar charts
    for ws in relevant_waits:
        # Filter the DataFrame for this wait-sharing value
        subdf_ws = df[df["WAIT_SHARING"] == ws]

        # -----------------------------------------------------
        # A) Survival Ratio Chart
        # -----------------------------------------------------
        # We'll compute survival ratio for Group A and B across FOOD_SHARING
        a_survival_ratios = []
        b_survival_ratios = []
        fs_labels = []  # store the FOOD_SHARING values as strings

        for fs_val in relevant_food_sharing:
            subdf_fs = subdf_ws[subdf_ws["FOOD_SHARING"] == fs_val]
            if len(subdf_fs) == 0:
                # If there's no data for that combination, append 0 or skip
                a_survival_ratios.append(0)
                b_survival_ratios.append(0)
                fs_labels.append(str(fs_val))
                continue

            a_win_count = subdf_fs["AWin"].sum()
            b_win_count = subdf_fs["BWin"].sum()
            group_size = len(subdf_fs)

            a_survival_ratios.append(a_win_count / group_size)
            b_survival_ratios.append(b_win_count / group_size)
            fs_labels.append(str(fs_val))

        # Create a bar chart with side-by-side bars for A and B
        x_positions = range(len(fs_labels))  # [0,1,2] for FS=1,2,3
        bar_width = 0.4

        plt.figure()
        plt.bar(
            [x - bar_width/2 for x in x_positions],
            a_survival_ratios,
            width=bar_width,
            label="Group A"
        )
        plt.bar(
            [x + bar_width/2 for x in x_positions],
            b_survival_ratios,
            width=bar_width,
            label="Group B"
        )

        plt.xticks(x_positions, fs_labels)
        plt.ylim(0, 1)
        plt.ylabel("Survival Ratio (Fraction of Runs Survived)")
        plt.xlabel("Food Sharing Threshold")
        plt.title(f"Survival Ratios (WAIT_SHARING={ws})")
        plt.legend()
        plt.grid(True)
        plt.show()

        # -----------------------------------------------------
        # B) Peak Population Chart
        # -----------------------------------------------------
        # We'll plot the *mean* PeakPopA vs. PeakPopB for each FOOD_SHARING
        mean_peak_a = []
        mean_peak_b = []

        for fs_val in relevant_food_sharing:
            subdf_fs = subdf_ws[subdf_ws["FOOD_SHARING"] == fs_val]
            if len(subdf_fs) == 0:
                mean_peak_a.append(0)
                mean_peak_b.append(0)
                continue

            # Compute average peak pop for A and B
            avg_a = subdf_fs["PeakPopA"].mean()
            avg_b = subdf_fs["PeakPopB"].mean()

            mean_peak_a.append(avg_a)
            mean_peak_b.append(avg_b)

        plt.figure()
        plt.bar(
            [x - bar_width/2 for x in x_positions],
            mean_peak_a,
            width=bar_width,
            label="PeakPopA"
        )
        plt.bar(
            [x + bar_width/2 for x in x_positions],
            mean_peak_b,
            width=bar_width,
            label="PeakPopB"
        )

        plt.xticks(x_positions, fs_labels)
        plt.ylabel("Average Peak Population")
        plt.title(f"Peak Population (WAIT_SHARING={ws})")
        plt.xlabel("Food Sharing Threshold")
        plt.legend()
        plt.grid(True)
        plt.show()

        # -----------------------------------------------------
        # C) Total Population Chart
        # -----------------------------------------------------
        # We'll plot the *mean* TotalPopA vs. TotalPopB for each FOOD_SHARING
        mean_total_a = []
        mean_total_b = []

        for fs_val in relevant_food_sharing:
            subdf_fs = subdf_ws[subdf_ws["FOOD_SHARING"] == fs_val]
            if len(subdf_fs) == 0:
                mean_total_a.append(0)
                mean_total_b.append(0)
                continue

            avg_a = subdf_fs["TotalPopA"].mean()
            avg_b = subdf_fs["TotalPopB"].mean()

            mean_total_a.append(avg_a)
            mean_total_b.append(avg_b)

        plt.figure()
        plt.bar(
            [x - bar_width/2 for x in x_positions],
            mean_total_a,
            width=bar_width,
            label="TotalPopA"
        )
        plt.bar(
            [x + bar_width/2 for x in x_positions],
            mean_total_b,
            width=bar_width,
            label="TotalPopB"
        )

        plt.xticks(x_positions, fs_labels)
        plt.ylabel("Average Total Population")
        plt.title(f"Total Population (WAIT_SHARING={ws})")
        plt.xlabel("Food Sharing Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("Done generating bar charts for WAIT_SHARING=0,100,200 and FOOD_SHARING=1,2,3.")

if __name__ == "__main__":
    main()
