import React, { useState } from "react";
import PopupBox from "./popupbox";
import Tesseract from "tesseract.js";
import { useMediaQuery } from "react-responsive";
import "./rightbox.css";

/*This file is to setup the box at the rightside of the page*/
/*Boxes at the right side of the page depend on the button clicked*/
const RightBox = ({ activeButton }) => {
  const [inputText, setInputText] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  const [responseMessage, setResponseMessage] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);

  // For viewports 1024px and below, apply smaller styling.
  const isTabletOrMobile = useMediaQuery({ maxWidth: 1024 });

  // Determine which container class to use:
  // On smaller devices, add both "center-box" and "small-box" classes.
  const containerClass = isTabletOrMobile ? "center-box small-box" : "right-box";

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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
  
      if (response.ok) {
        const data = await response.json();
        const chartData = {
          labels: data.sorted_sentiments.map((s) => s.sentiment),
          datasets: [
            {
              data: data.sorted_sentiments.map((s) => s.probability),
              backgroundColor: [
                "#FFC0CB", "#DC143C", "#00008B", "#ADD8E6", "#FFA500",
                "#3C6307", "#FFFF00", "#FF69B4", "#000000", "#808080",
                "#9F8C76", "#800080", "#9ACD32", "#FFD700", "#0000FF",
                "#FFFFFF", "#C0C0C0", "#4B0082"
              ],
            },
          ],
        };
        const message = JSON.stringify(chartData);
        setResponseMessage(message);
        setShowPopup(true);
      } else {
        setResponseMessage("Error: Unable to fetch sentiment analysis.");
        setShowPopup(true);
      }
    } catch (error) {
      setResponseMessage(`Error: ${error.message}`);
      setShowPopup(true);
    }
  };

  const handleImageSubmit = async (imageFile) => {
    try {
      const { data: { text } } = await Tesseract.recognize(
        imageFile,
        "eng",
        { logger: (info) => console.log(info) }
      );
      
      if (!text || !text.trim()) {
        alert("No readable text detected in the image.");
        return;
      }
  
      const response = await fetch("http://localhost:5000/analyze-image-sentiment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
  
      if (response.ok) {
        const data = await response.json();
        const chartData = {
          labels: data.sorted_sentiments.map((s) => s.sentiment),
          datasets: [
            {
              data: data.sorted_sentiments.map((s) => s.probability),
              backgroundColor: [
                "#FFC0CB", "#DC143C", "#00008B", "#ADD8E6", "#FFA500",
                "#3C6307", "#FFFF00", "#FF69B4", "#000000", "#808080",
                "#9F8C76", "#800080", "#9ACD32", "#FFD700", "#0000FF",
                "#FFFFFF", "#C0C0C0", "#4B0082"
              ],
            },
          ],
        };
        const message = JSON.stringify(chartData);
        setResponseMessage(message);
        setShowPopup(true);
      } else {
        alert("Error: Unable to analyze the image text.");
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className={containerClass}>
      <h2 className={isTabletOrMobile ? "small-font" : ""}>
        {titles[activeButton] || "Agent Info"}
      </h2>
      <div className={`dynamic-content ${activeButton === 1 && isTabletOrMobile ? "about-content" : ""}`}>
        {activeButton === 1 && (
          <p className={isTabletOrMobile ? "small-font" : ""}>
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
            <input
              type="file"
              accept="image/*"
              className="file-input"
              onChange={(event) => {
                const file = event.target.files[0];
                if (!file) {
                  alert("Please select an image file.");
                  return;
                }
                if (!file.type.startsWith("image/")) {
                  alert("Invalid file type. Please upload an image file.");
                  return;
                }
                setSelectedImage(URL.createObjectURL(file));
                console.log("Selected image:", file.name);
              }}
            />
            {selectedImage && (
              <div className="image-preview">
                <img src={selectedImage} alt="Selected" className="preview-image" />
              </div>
            )}
            <button
              className="submit-button"
              onClick={async () => {
                if (!selectedImage) {
                  alert("Please upload an image before submitting.");
                  return;
                }
                console.log("Submitting image:", selectedImage);
                await handleImageSubmit(selectedImage);
              }}
            >
              Submit
            </button>
          </div>
        )}
      </div>
      {showPopup && (
        <PopupBox
          message={responseMessage}
          onClose={() => setShowPopup(false)}
        />
      )}
    </div>
  );
};

export default RightBox;
