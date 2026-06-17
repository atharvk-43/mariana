# Telemetry Pipeline (EC2 Environment)

**Assignee:** DevOps Team (Teammate)
**Goal:** Ingest the network states and traffic flows into a central database.

## Architecture
We will use an industrial-grade but lightweight stack on the EC2 instance to gather data:
1. **Exporters:** Scraping the FRRouting containers.
2. **Collector:** Telegraf.
3. **Database:** InfluxDB (Port 8086 exposed for the ML team).

## Telemetry Sources

### 1. SNMP (Interface Metrics)
* Map the container `/sys/class/net/` directories to a host-level Prometheus SNMP Exporter, or install `snmpd` inside the containers.
* **Metrics:** Bytes In/Out, Packets, Drops, Errors.

### 2. NetFlow / IPFIX
* Run `fprobe` on the EC2 host, listening to the virtual `veth` interfaces created by Containerlab.
* Direct `fprobe` to send v5/v9 NetFlow records to Telegraf's UDP listener port `9995`.

### 3. Syslogs (BGP/OSPF Events)
* Configure FRRouting to log to `/var/log/frr/frr.log` (shared volume).
* Use `Fluent-Bit` or `Vector` to tail this file, extract keywords like `BGP-5-ADJCHANGE`, and send JSON payloads to InfluxDB.

### 4. Tunnel Statistics
* Write a small Python daemon on the EC2 host that runs `docker exec <ce-router> ipsec statusall` every 5 seconds.
* Parse the packet loss and rekey intervals and write them to InfluxDB.

## Deliverable
An InfluxDB instance accessible at `http://<EC2-IP>:8086`. The database `network_telemetry` should contain tables for `interfaces`, `flows`, `syslogs`, and `tunnels`.
