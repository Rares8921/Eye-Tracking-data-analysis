import pandas as pd
from utils import add_relative_time_columns
from utils import recordings, waldo_coords

margin = 0.0125

if __name__ == "__main__":
    for rec_id, rec_info in recordings.items():
        fix_file = f"neon_player_fixations/fixations_on_surface_Surface 1_{rec_id}.csv"

        df = pd.read_csv(fix_file)
        df = df.sort_values("start timestamp [ns]").reset_index(drop=True)
        add_relative_time_columns(df, start_col="start timestamp [ns]", end_col="end timestamp [ns]", duration_col="duration [ms]")

        output = []
        for level, start_s, end_s in rec_info:
            x_min, y_min, x_max, y_max = waldo_coords[level]
            x_min_p = max(0.0, x_min - margin)
            x_max_p = min(1.0, x_max + margin)
            y_min_p = max(0.0, y_min - margin)
            y_max_p = min(1.0, y_max + margin)

            subset = df[(df["start_s"] >= start_s) & (df["start_s"] <= end_s)]

            for _, row in subset.iterrows():
                x = row["fixation x [normalized]"]
                y = row["fixation y [normalized]"]
                dur = row["duration [ms]"]
                t = row["start_s"]

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    output.append([level, "direct", x, y, t, dur])
                elif x_min_p <= x <= x_max_p and y_min_p <= y <= y_max_p:
                    output.append([level, "peripheral", x, y, t, dur])

        out_df = pd.DataFrame(output, columns=["waldo", "type", "x", "y", "timestamp_s", "duration_ms"])
        out_file = f"waldo_fixations/waldo_fixations_{rec_id}.csv"

        if not out_df.empty:
            out_df.to_csv(out_file, index=False)