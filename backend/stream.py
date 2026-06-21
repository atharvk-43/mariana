import pandas as pd
import time
import os

from telemetry import generate_sample


while True:

    row = generate_sample()

    df = pd.DataFrame([row])

    file_exists = os.path.exists(
        "live_telemetry.csv"
    )

    df.to_csv(
        "backend/live_telemetry.csv",
        mode="a",
        header=not file_exists,
        index=False
    )

    time.sleep(2)