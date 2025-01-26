import React, { useState } from "react";
import PopupBox from "./popupbox"; // Import PopupBox component
import "./rightbox.css";

const RightBox = ({ activeButton }) => {
  const [inputText, setInputText] = useState(""); // State to track input text
  const [showPopup, setShowPopup] = useState(false); // State to toggle popup visibility
  const [responseMessage, setResponseMessage] = useState(""); // State to store backend response

  // Map button numbers to titles
  const titles = {
    1: "About",
    2: "Check Text",
    3: "Check Image",
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch("http://localhost:5000/analyze-sentiment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });
  
      if (response.ok) {
        const data = await response.json();
  
        // Extract the backend response data
        const finalSentiment = data.final_sentiment;
        const sortedSentiments = data.sorted_sentiments
          .map(
            (s) =>
              `${s.sentiment}: ${(s.probability * 100).toFixed(2)}%`
          )
          .join("\n");
  
        // Format the message for the PopupBox
        const message = `Final Sentiment: ${finalSentiment}\n\nDetailed Sentiments:\n${sortedSentiments}`;
        setResponseMessage(message); // Pass the message to the PopupBox
        setShowPopup(true); // Display the popup
      } else {
        setResponseMessage(`Error: Unable to fetch sentiment analysis.`);
        setShowPopup(true); // Display the error in the popup
      }
    } catch (error) {
      setResponseMessage(`Error: ${error.message}`);
      setShowPopup(true); // Display the error in the popup
    }
  };

  return (
    <div className="right-box">
      <h2>{titles[activeButton] || "Agent Info"}</h2>
      <div className="dynamic-content">
        {activeButton === 1 && (
          <p>
            Texted based communication like instant messaging, and emails are a
            convenient and rapid method of communication, but they lack a human
            element. In a typical conversation, there is eye contact, an audible
            voice, facial expressions and gestures. Each of these
            conversational elements are crucial in understanding a person's
            tone, emotion, and overall message behind the conversation. Because
            text-based communication lacks these crucial elements, it's often
            more difficult to understand a person's intended message through
            text. Therefore, we offer Sentimax. Sentimax is an AI-powered tool
            that semantically interprets a provided passage to predict the
            intended emotion behind the text. By predicting the emotion of a
            passage, Sentimax will help users better understand the intended
            message of the sender, allowing for clearer and more efficient
            communication.
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
            <button className="submit-button" onClick={handleSubmit}>
              Submit
            </button>
          </div>
        )}
        {activeButton === 3 && (
          <div className="center_image_button">
            <label className="upload-label">Upload an image:</label>
            <input type="file" accept="image/*" className="file-input" />
            <button className="submit-button">Submit</button>
          </div>
        )}
      </div>

      {/* Render the PopupBox */}
      {showPopup && (
        <PopupBox
          message={responseMessage}
          onClose={() => setShowPopup(false)} // Close the popup on button click
        />
      )}
    </div>
  );
};

export default RightBox;
