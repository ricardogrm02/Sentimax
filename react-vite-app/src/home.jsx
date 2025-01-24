import React, { useState } from 'react';
import './home.css';
import ThreeDModel from './components/ThreeDModel'; // Import the 3D model component
import SideButtons from './sidebutton.jsx';
import RightBox from './rightbox.jsx';

function App() {
  const [activeButton, setActiveButton] = useState(1); // Track which button is active

  return (
    <div>
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

      {/* Main Content */}
      <SideButtons setActiveButton={setActiveButton} />
      <RightBox activeButton={activeButton} />
    </div>
  );
}

export default App;
