import React, { useState } from "react";
import "./rightbox.css";

const RightBox = ({ activeButton }) => {
  const [inputText, setInputText] = useState(""); // State to track input text

  // Map button numbers to titles
  const titles = {
    1: "About",
    2: "Check Text",
    3: "Check Image",
  };

  return (
    <div className="right-box">
      <h2>{titles[activeButton] || "Agent Info"}</h2>
      <div className="dynamic-content">
        {activeButton === 1 && (
          <p>Kittens are small, fluffy, and absolutely adorable. They love to play and nap all day!</p>
        )}
        {activeButton === 2 && (
          <div className="text-input-section">
            <textarea
              className="text-input"
              placeholder="Enter your text here..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <button className="submit-button">Submit</button>
          </div>
        )}
        {activeButton === 3 && (
          <div className="center_image_button">
            <label className="upload-label">Upload an image:</label>
            <input type="file" accept="image/*" className="file-input" />
          </div>
        )}
      </div>
    </div>
  );
};

export default RightBox;
