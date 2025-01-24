import React, { useState, useEffect } from 'react';
import './RightBox.css';

const RightBox = ({ activeButton }) => {
  const [content, setContent] = useState(null); // State to manage the dynamic content

  // Update content based on the active button
  useEffect(() => {
    if (activeButton === 1) {
      setContent(<p>Kittens are small, fluffy, and absolutely adorable. They love to play and nap all day!</p>);
    } else if (activeButton === 2) {
      setContent(
        <textarea
          placeholder="Enter text here..."
          className="text-input"
        />
      );
    } else if (activeButton === 3) {
      setContent(
        <div>
          <label className="upload-label">Upload an image:</label>
          <input type="file" accept="image/*" className="file-input" />
        </div>
      );
    }
  }, [activeButton]);

  return (
    <div className="right-box">
      <h2>Agent Info</h2>
      <div className="dynamic-content">{content}</div>
    </div>
  );
};

export default RightBox;