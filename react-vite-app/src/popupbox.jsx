import React from "react";
import "./popupbox.css";

const PopupBox = ({ message, onClose }) => {
  return (
    <div className="popup-box-overlay">
      <div className="popup-box">
        <p>{message}</p>
        <button className="close-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default PopupBox;
