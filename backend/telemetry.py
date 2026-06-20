import random
import pandas as pd
from datetime import datetime

def generate_sample():

    return {

        "timestamp": datetime.utcnow().isoformat(),

        "node_id": random.choice([
            "PE-1",
            "PE-2",
            "P-1",
            "CE-1",
            "DC-1"
        ]),

        "cpu_load_pct":
            round(random.uniform(10, 95), 2),

        "memory_used_pct":
            round(random.uniform(20, 90), 2),

        "utilization_pct":
            round(random.uniform(5, 95), 2),

        "latency_ms":
            round(random.uniform(1, 120), 2),

        "jitter_ms":
            round(random.uniform(0.1, 20), 2),

        "packet_loss_pct":
            round(random.uniform(0, 5), 2),

        "bgp_sessions_active":
            random.randint(1, 5),

        "bgp_updates_per_min":
            random.randint(0, 50),

        "queue_depth":
            random.randint(0, 100)

    }

def save_samples(count=1000):

    data = [
        generate_sample()
        for _ in range(count)
    ]

    df = pd.DataFrame(data)

    df.to_csv(
        "telemetry.csv",
        index=False
    )

if __name__ == "__main__":
    save_samples()