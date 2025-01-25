import React, { useState } from 'react';
import './home.css';
import SideButtons from './sidebutton.jsx';
import RightBox from './rightbox.jsx';

function App() {
  const [activeButton, setActiveButton] = useState(1); // Track which button is active

  const getImage = () => {
    switch (activeButton) {
      case 1:
        return 'Firefly_40_EDIT-Photoroom.png'; // About button
      case 2:
        return 'Firefly_47-Photoroom.png'; // Check Text button
      case 3:
        return 'Firefly_44-Photoroom.png'; // Check Image button
      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      {/* Background Video */}
      <video
        className="background-video"
        autoPlay
        muted
        loop
        playsInline
      >
        <source src="/anime-girl-watching-sunset-by-cherry-tree-moewalls-com.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      <div className="content-layout">
        {/* Side Buttons */}
        <SideButtons setActiveButton={setActiveButton} />

        {/* Display Image Based on Button */}
        <div className="image-container">
          {getImage() && (
            <img src={getImage()} alt="Active Content" className="dynamic-image" />
          )}
        </div>

        {/* Right Box */}
        <RightBox activeButton={activeButton} />
      </div>
    </div>
  );
}

export default App;
