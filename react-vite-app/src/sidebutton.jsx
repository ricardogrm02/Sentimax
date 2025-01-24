import React, { useState } from 'react';
import './sidebutton.css';

const SideButtons = ({ setActiveButton }) => {
  return (
    <div className="side-buttons">
      <button className="circle-button" onClick={() => setActiveButton(1)}>1</button>
      <button className="circle-button" onClick={() => setActiveButton(2)}>2</button>
      <button className="circle-button" onClick={() => setActiveButton(3)}>3</button>
    </div>
  );
};

export default SideButtons;