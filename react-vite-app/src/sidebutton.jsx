import React, { useState } from 'react';
import './sidebutton.css';

const SideButtons = ({ setActiveButton }) => {
  return (
    <div className="side-buttons">
      <button className="circle-button" onClick={() => setActiveButton(1)}>About</button>
      <button className="circle-button" onClick={() => setActiveButton(2)}>Check Text</button>
      <button className="circle-button" onClick={() => setActiveButton(3)}>Check Image</button>
    </div>
  );
};

export default SideButtons;