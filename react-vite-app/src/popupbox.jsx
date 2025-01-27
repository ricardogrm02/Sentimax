import React from "react";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";
import "./popupbox.css";

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PopupBox = ({ message, onClose }) => {
  let chartData;

  try {
    // Parse the JSON message for chart data
    chartData = JSON.parse(message);
  } catch (e) {
    chartData = null; // If parsing fails, default to no chart data
  }

  return (
    <div className="popup-box-overlay">
      <div className="popup-box">
        {chartData ? (
          <Pie data={chartData} />
        ) : (
          <p>{message}</p>
        )}
        <button className="close-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default PopupBox;
