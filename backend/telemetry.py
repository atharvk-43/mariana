import random
import pandas as pd
from datetime import datetime

def generate_sample():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "battery_voltage": round(random.uniform(26, 30), 2),
        "temperature": round(random.uniform(-20, 60), 2),
        "cpu_load": round(random.uniform(10, 90), 2),
        "signal_strength": round(random.uniform(50, 100), 2)
    }

def save_samples(count=1000):
    data = [generate_sample() for _ in range(count)]
    df = pd.DataFrame(data)
    df.to_csv("telemetry.csv", index=False)

if __name__ == "__main__":
    save_samples()
