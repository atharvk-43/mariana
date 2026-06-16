async function loadData() {

    try {

        const latestResponse =
        await fetch("http://localhost:8000/latest");

        const telemetry =
        await latestResponse.json();

        document.getElementById("battery").innerText =
        telemetry.battery_voltage;

        document.getElementById("temperature").innerText =
        telemetry.temperature;

        document.getElementById("cpu").innerText =
        telemetry.cpu_load;

        document.getElementById("signal").innerText =
        telemetry.signal_strength;

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
