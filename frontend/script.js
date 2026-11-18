// NAVIGATION
const navItems = document.querySelectorAll(".nav-item");
const pageTitle = document.getElementById("page-title");

navItems.forEach((button) => {
    button.addEventListener("click", () => {
        navItems.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");

        const page = button.dataset.page;
        pageTitle.textContent =
            page === "Dashboard"
                ? "Welcome to Solar Power Analytics"
                : page;
    });
});

// ACTUAL VS. PREDICTED GENERATION CHART
const generationCanvas = document.getElementById("generationChart");

new Chart(generationCanvas, {
    type: "line",
    data: {
        labels: [
            "8 AM", "9 AM", "10 AM", "11 AM", "12 PM",
            "1 PM", "2 PM", "3 PM", "4 PM", "5 PM"
        ],
        datasets: [
            {
                label: "Actual Solar Power Generation",
                data: [24, 43, 66, 85, 102, 110, 91, 82, 56, 29],
                borderColor: "#1F3D2B",
                backgroundColor: "#1F3D2B",
                borderWidth: 2.5,
                tension: 0.3,
                pointRadius: 3
            },
            {
                label: "Predicted Solar Power Generation",
                data: [28, 47, 63, 87, 99, 106, 103, 79, 58, 31],
                borderColor: "#8B5CF6",
                backgroundColor: "#8B5CF6",
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.3,
                pointRadius: 3
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom",
                labels: {
                    boxWidth: 12,
                    font: {
                        size: 10
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: "kWh"
                }
            }
        }
    }
});

// POWER GENERATION BAR CHART
const powerCanvas = document.getElementById("powerChart");

new Chart(powerCanvas, {
    type: "bar",
    data: {
        labels: [
            "8 AM", "9 AM", "10 AM", "11 AM", "12 PM",
            "1 PM", "2 PM", "3 PM", "4 PM", "5 PM"
        ],
        datasets: [
            {
                label: "Power Generation",
                data: [23, 42, 65, 83, 102, 110, 90, 82, 54, 28],
                backgroundColor: "#1F3D2B",
                borderRadius: 4,
                barPercentage: 0.7
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: "kWh"
                }
            }
        }
    }
});

// BACKEND POWER PREDICTION
const API_BASE = "https://solar-api-876m.onrender.com";

const predictionForm = document.getElementById("prediction-form");
const predictionStatus = document.getElementById("prediction-status");
const predictionResult = document.getElementById("prediction-result");
const predictedPowerValue = document.getElementById("predicted-power-value");
const predictionButton = predictionForm.querySelector('button[type="submit"]');

predictionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    predictionStatus.textContent = "Requesting prediction…";
    predictionResult.hidden = true;
    predictionButton.disabled = true;

    const requestData = {
        plant_id: Number(document.getElementById("plant-id").value),
        date_time: document.getElementById("date-time").value.trim(),
        ambient_temperature: Number(
            document.getElementById("ambient-temperature").value
        ),
        module_temperature: Number(
            document.getElementById("module-temperature").value
        ),
        irradiation: Number(document.getElementById("irradiation").value)
    };

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `Request failed (${response.status})`);
        }

        predictedPowerValue.textContent =
            `${data.predicted_ac_power_kw} kW`;
        predictionResult.hidden = false;
        predictionStatus.textContent = "Prediction received from the backend.";
    } catch (error) {
        predictionStatus.textContent =
            `Could not get a prediction: ${error.message}. ` +
            `Check that the backend is running at ${API_BASE}.`;
    } finally {
        predictionButton.disabled = false;
    }
});