# Network Simulator (EC2 Environment)

**Assignee:** Networking / DevOps Team (Teammate)
**Environment:** 8 vCPU, 16 GiB RAM EC2 Instance
**Tech Stack:** Docker, Containerlab, FRRouting (FRR)

## Topology Design
Deploy a 6-node network using a `topology.yml` file in Containerlab:
1. `PE-Hub`: Provider Edge at the Datacenter.
2. `PE-Branch-1` & `PE-Branch-2`: Provider Edges at branches.
3. `P-Core`: Provider transit router.
4. `CE-Hub`, `CE-Branch-1`, `CE-Branch-2`: Customer Edge routers.

## Routing & Underlay (MPLS/OSPF)
* Configure **OSPF** across `PE-Hub`, `P-Core`, and the branch `PE` routers to establish the core network.
* Enable **LDP (Label Distribution Protocol)** on these core links to satisfy the MPLS problem statement requirement.

## Overlay & Tunnels (SD-WAN / BGP)
* Configure **eBGP** peering between the `CE` routers and their respective `PE` routers.
* Establish **GRE-over-IPSec** tunnels (using `ipsec` or `wireguard`) directly between `CE-Branch-1/2` and `CE-Hub`.
* Route customer application traffic through these tunnels.

## Traffic Generation
* Run `iperf3` daemon on `CE-Hub`.
* Run background cron jobs on the branch CE routers to send continuous varying TCP/UDP traffic to the hub to simulate diurnal enterprise network loads.

## Fault Injection Scripts (The Ground Truth)
Write a bash script `inject_faults.sh` with the following functions:
1. **Scenario 1 (Congestion):** Use `tc qdisc add dev eth1 root tbf rate 1mbit burst 10kbit` on a core link to artificially choke bandwidth.
2. **Scenario 2 (Route Flap):** Loop `ip link set dev eth1 down` and `up` every 30 seconds to force BGP/OSPF recalculations.
3. **Scenario 3 (Tunnel Degradation):** Use `tc netem loss 5% delay 100ms` on the tunnel interfaces to simulate jitter and loss.
