import random
import pandas as pd
from datetime import datetime, UTC

NODES = {
    "PE-1": "PE",
    "PE-2": "PE",
    "P-1": "P",
    "CE-1": "CE",
    "DC-1": "DC"
}

INTERFACES = [
    "eth0",
    "eth1",
    "eth2"
]


def generate_sample():

    node_id = random.choice(list(NODES.keys()))
    node_type = NODES[node_id]

    # Node-type aware utilization
    if node_type == "CE":
        utilization = round(random.uniform(5, 50), 2)

    elif node_type == "PE":
        utilization = round(random.uniform(30, 90), 2)

    elif node_type == "P":
        utilization = round(random.uniform(40, 95), 2)

    else:  # DC
        utilization = round(random.uniform(50, 95), 2)

    # Traffic volume
    bytes_in = int(utilization * random.randint(50000, 120000))
    bytes_out = int(bytes_in * random.uniform(0.7, 1.3))

    packets_in = max(
        100,
        int(bytes_in / random.randint(600, 1200))
    )

    packets_out = max(
        100,
        int(bytes_out / random.randint(600, 1200))
    )

    # Congestion indicators
    queue_depth = int(
        utilization * random.uniform(0.5, 1.2)
    )

    latency_ms = round(
        5 +
        utilization * 0.8 +
        random.uniform(-3, 3),
        2
    )

    packet_loss_pct = round(
        max(
            0,
            (utilization - 70) / 20
        ) * random.uniform(0, 1),
        2
    )

    jitter_ms = round(
        latency_ms * random.uniform(0.05, 0.15),
        2
    )

    # Device load
    cpu_load_pct = round(
        min(
            95,
            utilization * random.uniform(0.6, 1.1)
        ),
        2
    )

    memory_used_pct = round(
        min(
            90,
            20 + utilization * random.uniform(0.3, 0.7)
        ),
        2
    )

    # Routing activity
    bgp_updates = random.choices(
        [0, 1, 2, 3, 5, 10],
        weights=[40, 25, 15, 10, 7, 3],
        k=1
    )[0]

    bgp_withdrawals = random.choices(
        [0, 1, 2, 3],
        weights=[80, 15, 4, 1],
        k=1
    )[0]

    return {

        # Identity
        "timestamp": datetime.now(UTC).isoformat(),
        "node_id": node_id,
        "node_type": node_type,
        "interface": random.choice(INTERFACES),

        # Interface Metrics
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "packets_in": packets_in,
        "packets_out": packets_out,

        "errors_in": random.choices(
            [0, 1, 2, 3],
            weights=[85, 10, 4, 1],
            k=1
        )[0],

        "drops_in": int(packet_loss_pct * random.uniform(0, 20)),
        "drops_out": int(packet_loss_pct * random.uniform(0, 20)),

        "utilization_pct": utilization,

        "link_state": random.choices(
            ["UP", "FLAPPING", "DOWN"],
            weights=[97, 2, 1],
            k=1
        )[0],

        # Routing
        "bgp_sessions_active": random.randint(1, 5),
        "bgp_prefixes_received": random.randint(100, 500),
        "bgp_updates_per_min": bgp_updates,
        "bgp_withdrawals_per_min": bgp_withdrawals,

        "ospf_spf_runs": random.choices(
            [0, 1, 2, 3],
            weights=[70, 20, 8, 2],
            k=1
        )[0],

        # System
        "cpu_load_pct": cpu_load_pct,
        "memory_used_pct": memory_used_pct,
        "queue_depth": queue_depth,

        # Performance
        "latency_ms": latency_ms,
        "jitter_ms": jitter_ms,
        "packet_loss_pct": packet_loss_pct,

        # Tunnel/IPSec
        "tunnel_packet_loss_pct": round(
            packet_loss_pct * random.uniform(0.5, 1.5),
            2
        ),

        "ipsec_rekeyed_last_hr": random.choices(
            [0, 1, 2],
            weights=[20, 70, 10],
            k=1
        )[0]
    }


def save_samples(count=1000):

    data = [
        generate_sample()
        for _ in range(count)
    ]

    df = pd.DataFrame(data)

    df.to_csv(
        "backend/telemetry.csv",
        index=False
    )

    print(
        f"Generated {count} telemetry records and saved to backend/telemetry.csv"
    )


if __name__ == "__main__":
    save_samples()