async function loadData() {

    try {

        const latestResponse =
        await fetch("http://localhost:8000/latest");

        const telemetry =
        await latestResponse.json();

        document.getElementById("node").innerText =
telemetry.node_id;

document.getElementById("cpu").innerText =
telemetry.cpu_load_pct;

document.getElementById("memory").innerText =
telemetry.memory_used_pct;

document.getElementById("latency").innerText =
telemetry.latency_ms;

document.getElementById("loss").innerText =
telemetry.packet_loss_pct;

        const anomalyResponse =
        await fetch("http://localhost:8000/anomalies");

        const stats =
        await anomalyResponse.json();

        document.getElementById("records").innerText =
        stats.total_records;

        document.getElementById("anomalies").innerText =
        stats.anomalies_detected;

    }
    catch(error) {

        console.log(error);

    }
}

loadData();

setInterval(loadData, 2000);
