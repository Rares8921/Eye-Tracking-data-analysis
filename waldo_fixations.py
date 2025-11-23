import pandas as pd
import os

intervale = {
    "2025-09-26_16-24-13-61db42a9": [
        ("waldo_3", 0.000, 60.785),
        ("waldo_4", 60.892, 96.252)
    ],
    "2025-09-26_16-45-00-95032692": [
        ("waldo_1", 0.000, 98.074),
        ("waldo_2", 101.917, 112.556),
        ("waldo_3", 116.497, 121.522),
        ("waldo_4", 123.295, 137.777)
    ],
    "2025-09-26_19-01-53-2ec86210": [
        ("waldo_1", 0.000, 311.217),
        ("waldo_2", 314.024, 325.354),
        ("waldo_3", 329.393, 419.438),
        ("waldo_4", 423.477, 451.358)
    ],
    "2025-09-26_19-10-19-05411873": [
        ("waldo_1", 6.995, 86.055),
        ("waldo_2", 87.976, 91.868),
        ("waldo_3", 95.611, 99.404),
        ("waldo_4", 102.901, 118.861)
    ],
    "2025-09-26_19-12-44-115b6a40": [
        ("waldo_1", 8.340, 136.545),
        ("waldo_2", 138.663, 140.240),
        ("waldo_3", 144.525, 189.794),
        ("waldo_4", 192.306, 193.735)
    ],
    "2025-09-26_19-17-52-c0419fea": [
        ("waldo_1", 0.000, 17.733),
        ("waldo_2", 18.226, 35.368),
        ("waldo_3", 38.964, 62.460),
        ("waldo_4", 66.499, 69.209)
    ],
    "2025-09-26_19-27-31-2699d4a1": [
        ("waldo_1", 0.000, 60.736),
        ("waldo_2", 62.608, 76.203),
        ("waldo_3", 80.735, 97.040),
        ("waldo_4", 99.503, 107.187)
    ],
    "2025-09-26_19-29-58-b4f7a1df": [
        ("waldo_1", 15.467, 130.388),
        ("waldo_2", 132.752, 136.102),
        ("waldo_3", 140.190, 146.348),
        ("waldo_4", 150.436, 152.801)
    ],
    "2025-09-26_19-37-31-91230d39": [
        ("waldo_1", 31.526, 64.775),
        ("waldo_2", 66.844, 68.962),
        ("waldo_3", 71.031, 77.632),
        ("waldo_4", 81.178, 84.429)
    ],
    "2025-09-26_19-39-21-36589e44": [
        ("waldo_1", 5.369, 148.860),
        ("waldo_2", 155.608, 189.400),
        ("waldo_3", 193.390, 349.638),
        ("waldo_4", 352.791, 443.821)
    ],
    "2025-09-26_20-20-21-410567f0": [
        ("waldo_1", 21.772, 117.679),
        ("waldo_2", 120.339, 124.428),
        ("waldo_3", 126.841, 159.549),
        ("waldo_4", 163.145, 255.012)
    ]
}

waldo_coords = {
    "waldo_1": (931/1979, 375/1079, 974/1979, 433/1079),
    "waldo_2": (1197/1979, 186/1079, 1277/1979, 382/1079),
    "waldo_3": (446/1979, 204/1079, 484/1979, 263/1079),
    "waldo_4": (1081/1979, 306/1079, 1107/1979, 397/1079)
}

margin = 0.0125

for rec_id, aoi_list in intervale.items():
    fix_file = f"neon_player_fixations/fixations_on_surface_Surface 1_{rec_id}.csv"
    if not os.path.exists(fix_file):
        print(f"{fix_file} nu exista")
        continue

    df = pd.read_csv(fix_file)
    df = df.sort_values("start timestamp [ns]").reset_index(drop=True)
    df["start_s"] = (df["start timestamp [ns]"] - df["start timestamp [ns]"].iloc[0]) / 1e9
    df["end_s"] = (df["end timestamp [ns]"] - df["start timestamp [ns]"].iloc[0]) / 1e9

    output_rows = []

    for waldo, start_s, end_s in aoi_list:
        x_min, y_min, x_max, y_max = waldo_coords[waldo]
        x_min_p = max(0, x_min - margin)
        x_max_p = min(1, x_max + margin)
        y_min_p = max(0, y_min - margin)
        y_max_p = min(1, y_max + margin)

        subset = df[(df["start_s"] >= start_s) & (df["start_s"] <= end_s)]

        for _, row in subset.iterrows():
            x = row["fixation x [normalized]"]
            y = row["fixation y [normalized]"]
            dur = row["duration [ms]"]
            t = row["start_s"]

            if x_min <= x <= x_max and y_min <= y <= y_max:
                output_rows.append([waldo, "direct", x, y, t, dur])
            elif x_min_p <= x <= x_max_p and y_min_p <= y <= y_max_p:
                output_rows.append([waldo, "peripheral", x, y, t, dur])

    if output_rows:
        out_df = pd.DataFrame(output_rows, columns=["waldo", "type", "x", "y", "timestamp_s", "duration_ms"])
        out_file = f"waldo_fixations/waldo_fixations_{rec_id}.csv"
        out_df.to_csv(out_file, index=False)
