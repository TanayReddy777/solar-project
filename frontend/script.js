// NAVIGATION

const navItems = document.querySelectorAll(".nav-item");

const pageTitle = document.getElementById("page-title");


navItems.forEach(function(button) {

    button.addEventListener("click", function() {

        navItems.forEach(function(item) {

            item.classList.remove("active");

        });

        button.classList.add("active");

        const page = button.dataset.page;

        if (page === "Dashboard") {

            pageTitle.textContent = "Welcome to Solar Power Analytics";

        } else {

            pageTitle.textContent = page;

        }

    });

});


// ACTUAL VS PREDICTED GENERATION CHART

const generationCanvas = document.getElementById("generationChart");

new Chart(generationCanvas, {

    type: "line",

    data: {

        labels: [
            "8 AM",
            "9 AM",
            "10 AM",
            "11 AM",
            "12 PM",
            "1 PM",
            "2 PM",
            "3 PM",
            "4 PM",
            "5 PM"
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

            "8 AM",
            "9 AM",
            "10 AM",
            "11 AM",
            "12 PM",
            "1 PM",
            "2 PM",
            "3 PM",
            "4 PM",
            "5 PM"

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