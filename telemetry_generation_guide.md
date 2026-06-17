# Rich Telemetry Generation Guide

With access to an **8 vCPU, 16 GiB RAM EC2 instance**, we have ample resources to deploy a professional network telemetry collection stack. Below is the exact technical blueprint for how we can implement and capture all five required data sources in our Containerlab/FRRouting simulation.

---

## 1. How to Capture the 5 Data Sources

### 1.1 SNMP (Interface & Device Metrics)
* **What it is:** Hardware/interface status, packets/bytes in/out, error counters, drops.
* **Implementation in Containerlab:**
  1. We install `snmpd` (the SNMP daemon) inside each FRRouting container during the container startup script, or build a custom Dockerfile:
     ```dockerfile
     FROM frrouting/frr:latest
     RUN apt-get update && apt-get install -y snmpd
     ```
  2. Map a custom `/etc/snmp/snmpd.conf` allowing read-only access to standard MIBs.
  3. Spin up a **Prometheus SNMP Exporter** container. It will scrape each router container via SNMP GET requests on port `161` and convert them into Prometheus metrics.

---

### 1.2 NetFlow / IPFIX (Traffic Flows)
* **What it is:** Packet metadata (source/destination IP, ports, protocols, byte size) to analyze application congestion.
* **Implementation in Containerlab:**
  1. Since Containerlab links are standard Linux `veth` pairs on the EC2 host, we do not need to install heavy exporters on every router.
  2. We run **`fprobe`** (a lightweight NetFlow probe) on the EC2 host for each virtual interface corresponding to a container link:
     ```bash
     # Install fprobe on host
     sudo apt-get install -y fprobe
     # Capture traffic on a specific container link and export it to collector on port 9995
     sudo fprobe -i clab-isro-mpls-network-pe-router-1-eth1 -fip 127.0.0.1:9995
     ```
  3. The flows are sent to a collector container running **`goflow2`** or **`Logstash`**, which outputs structured JSON.

---

### 1.3 Syslogs (Routing Protocol Events)
* **What it is:** Immediate text logs generated when BGP sessions drop, OSPF links flap, or interfaces change state.
* **Implementation in Containerlab:**
  1. FRRouting uses `syslog` to output routing daemon events. We configure `/etc/frr/support_bundle.conf` or `frr.conf` to direct logs to syslog:
     ```text
     log syslog informational
     log file /var/log/frr/frr.log informational
     ```
  2. We mount the `/var/log/frr/` directory of each router container as a shared volume to the EC2 host.
  3. We run a lightweight collector container like **Fluent-Bit** or **Vector** to monitor the log files, parse strings using regular expressions, and extract events:
     * *Example pattern to parse:* `BGP-5-ADJCHANGE: neighbor 10.0.1.2 Down`

---

### 1.4 Streaming Telemetry (gNMI / CLI Engine)
* **What it is:** High-frequency push-based telemetry.
* **Implementation in Containerlab:**
  1. Real-world gNMI requires commercial OS support (like Cisco IOS-XR or Arista EOS). In a lightweight FRR stack, we achieve the same telemetry resolution by querying the routing engine's UNIX socket via `vtysh` and outputting JSON.
  2. We run a Python telemetry daemon that executes CLI commands directly on the docker sockets of the containers:
     ```python
     import json
     import subprocess

     # Execute CLI query inside router container
     result = subprocess.check_output([
         "docker", "exec", "clab-isro-mpls-network-dc-router", 
         "vtysh", "-c", "show ip bgp summary json"
     ])
     bgp_stats = json.loads(result)
     ```
  3. The Python collector publishes this parsed state data directly to a FastAPI server or InfluxDB every 1-2 seconds.

---

### 1.5 Tunnel Statistics (IPSec / GRE overlays)
* **What it is:** Packet loss, latency, jitter, and key exchange statistics across the SD-WAN overlay.
* **Implementation in Containerlab:**
  1. We establish GRE tunnels inside the FRR containers and secure them using `StrongSwan` (IPSec).
  2. To get tunnel telemetry:
     * **Throughput:** Scraped from `/sys/class/net/gre0/statistics/rx_packets` inside the container namespace.
     * **Jitter & Latency:** We run active probes (e.g., sending small UDP ping packets every 200ms) across the tunnels.
     * **IPSec State:** Run `docker exec router strongswan statusall` and parse the output to detect rekey timings and connection status.

---

## 2. Integrated Telemetry Architecture on EC2

To tie everything together on the 16GB EC2 instance, we will deploy the following pipeline:

```mermaid
graph TD
    subgraph Network Nodes (Docker Containers)
        R1[FRR Node 1]
        R2[FRR Node 2]
    end

    subgraph Data Exporters
        R1 -->|Syslogs| LogVol[Shared Volume /var/log/frr]
        R1 -->|SNMP port 161| SNMP_Exp[SNMP Exporter]
        Veth[Host veth Interfaces] -->|pcap| Fprobe[fprobe NetFlow daemon]
    end

    subgraph Collection & Storage (Docker Compose)
        Vector[Vector Log Collector] <-- Scrapes --> LogVol
        Telegraf[Telegraf Agent] <-- Pulls SNMP --> SNMP_Exp
        Telegraf <-- Listens UDP --> Fprobe
        
        Vector -->|Structured JSON| DB[(InfluxDB / TimescaleDB)]
        Telegraf -->|Time-Series Metrics| DB
    end

    subgraph ML Pipeline
        DB -->|Query Historical Window| ML[FastAPI ML Engine]
        ML -->|Predictive Scoring| Dashboard[Web UI]
    end
```

---

## 3. How to Setup This Rich Telemetry Pipeline

### Step 1: Install Exporters on the EC2 Instance
First, install the collection tools on the EC2 host:
```bash
sudo apt-get update
sudo apt-get install -y fprobe iptables
```

### Step 2: Define Docker-Compose for Storage & Collection
Create a `docker-compose.yml` to spin up InfluxDB (or Prometheus) and Telegraf:
```yaml
version: '3'
services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2

  telegraf:
    image: telegraf:latest
    volumes:
      - ./telegraf.conf:/etc/telegraf/telegraf.conf:ro
    ports:
      - "9995:9995/udp" # NetFlow port
    depends_on:
      - influxdb

volumes:
  influxdb_data:
```

### Step 3: Run the Telegraf Collector Config
Configure Telegraf to accept NetFlow records on port `9995` and write to InfluxDB:
```toml
[[inputs.netflow]]
  interface = "127.0.0.1:9995"
  version = "v5"

[[outputs.influxdb_v2]]
  urls = ["http://influxdb:8086"]
  token = "$INFLUX_TOKEN"
  organization = "isro"
  bucket = "network_telemetry"
```
