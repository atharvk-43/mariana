import pandas as pd
import random
import time
from datetime import datetime

while True:

    row = {
        "timestamp": datetime.now().isoformat(),
        "battery_voltage": round(random.uniform(27,30),2),
        "temperature": round(random.uniform(-20,35),2),
        "cpu_load": round(random.uniform(10,90),2),
        "signal_strength": round(random.uniform(50,80),2)
    }

    df = pd.DataFrame([row])

    try:
        df.to_csv(
            "live_telemetry.csv",
            mode="a",
            header=not pd.io.common.file_exists("live_telemetry.csv"),
            index=False
        )
    except:
        pass

    time.sleep(2)
