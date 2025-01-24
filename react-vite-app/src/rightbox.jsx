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
          <p>
            Texted based communication like instant messaging, and emails are a convenient and rapid method of communication, but they lacks a human element. 
            In a typical conversation, there is eye contact, an audible voice, facial expressions and gestures. 
            Each of these conversational elements are crucial in understanding a person's tone, emotion, and overall message behind the conversation. 
            Because text based communication lacks these crucial elements, it's often more difficult to understand a persons intended message through a text. 
            Therefore, we offer Sentimax. Sentimax is an AI powered tool that semantically interprets a provided passage in order to predict the intended emotion behind the text. 
            By predicting the emotion of a passage, Sentimax will help users better understand the intended message of the sender, 
            allowing for clearer and more efficient communication.
          </p>
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
