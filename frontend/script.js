alert("SCRIPT LOADED");

async function loadData() {

    try {

        const latestResponse =
        await fetch("http://13.217.137.172:8000/latest");

        alert("LATEST STATUS = " + latestResponse.status);

        const telemetry =
        await latestResponse.json();

        alert("BATTERY = " + telemetry.battery_voltage);

        document.getElementById("battery").innerText =
        telemetry.battery_voltage;

        document.getElementById("temperature").innerText =
        telemetry.temperature;

        document.getElementById("cpu").innerText =
        telemetry.cpu_load;

        document.getElementById("signal").innerText =
        telemetry.signal_strength;

        const anomalyResponse =
        await fetch("http://13.217.137.172:8000/anomalies");

        alert("ANOMALY STATUS = " + anomalyResponse.status);

        const stats =
        await anomalyResponse.json();

        document.getElementById("records").innerText =
        stats.total_records;

        document.getElementById("anomalies").innerText =
        stats.anomalies_detected;

    }
    catch(error) {

        alert("ERROR = " + error);

    }
}

loadData();