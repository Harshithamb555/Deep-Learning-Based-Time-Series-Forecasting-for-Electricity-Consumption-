let chart;

async function makePrediction() {
    const city = document.getElementById("cityInput").value;
    const modelName = document.getElementById("modelSelect").value;

    if (!city) {
        alert("Please enter a city name!");
        return;
    }

    // Fake history because you will connect real weather API later
    let history = [23, 24, 22, 26, 25, 27, 28]; // 7-day sample

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            history: history,
            model: modelName
        })
    });

    const result = await response.json();

    drawChart(result.prediction);
}

function drawChart(values) {
    const ctx = document.getElementById("forecastChart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ["Day 1","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7"],
            datasets: [{
                label: "Predicted Temperature",
                data: values,
                borderColor: "#0078ff",
                backgroundColor: "rgba(0,120,255,0.2)",
                fill: true,
                tension: 0.4,
                borderWidth: 3
            }]
        }
    });
}
