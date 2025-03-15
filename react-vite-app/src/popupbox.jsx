import React from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import "./popupbox.css";

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PopupBox = ({ message, onClose }) => {
  let chartData;
  try {
    // Parse the JSON message for chart data
    chartData = JSON.parse(message);
  } catch (e) {
    chartData = null;
  }

  // Define options to force the legend labels to use white text
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        labels: {
          color: "white",
          font: {
            size: 12,
            family: "Arial, sans-serif",
          },
        },
      },
    },
  };

  return (
    <div className="popup-box-overlay">
      <div className="popup-box">
        {/* New wrapper for the Pie chart */}
        <div className="pie-chart-wrapper">
          {chartData ? (
            <Pie data={chartData} options={chartOptions} />
          ) : (
            <p>{message}</p>
          )}
        </div>
        <button className="close-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default PopupBox;
