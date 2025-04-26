import React, { useState, useRef, useEffect } from 'react';
import './home.css';
import SideButtons from './sidebutton.jsx';
import RightBox from './rightbox.jsx';

/* This file is to setup the homepage and its assets */
/* This file loads in the interactive rightbox & sidebuttons on the left */
function App() {
  const [activeButton, setActiveButton] = useState(1); // Track which button is active
  const audioRef = useRef(null); // Reference for the background music

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = 0.1; // Set volume to 50%
      audioRef.current.play().catch((err) => {
        console.log("Autoplay blocked, waiting for user interaction...");
      });
    }
  }, []);

  const getImage = () => {
    switch (activeButton) {
      case 1:
        return 'Firefly_40_EDIT-Photoroom.png'; // About button
      case 2:
        return 'Firefly_47_EDIT-Photoroom.png'; // Check Text button
      case 3:
        return 'Firefly_44_EDIT-Photoroom.png'; // Check Image button
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

      {/* Background Music */}
      <audio
        ref={audioRef}
        src="/Cherry Blossom Festival.mp3"
        loop
        preload="auto"
      />

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
