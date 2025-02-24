import React, { useState } from 'react';
import { useMediaQuery } from 'react-responsive';
import './sidebutton.css';

/*This file is to setup buttons that control the page state*/
const SideButtons = ({ setActiveButton }) => {
  // For viewports 1023px wide and below, we consider it tablet or smaller.
  const isTabletOrSmaller = useMediaQuery({ maxWidth: 1024 });

  return (
    <div className={isTabletOrSmaller ? "top-buttons" : "side-buttons"}>
      <button className="circle-button" onClick={() => setActiveButton(1)}>About</button>
      <button className="circle-button" onClick={() => setActiveButton(2)}>Check Text</button>
      <button className="circle-button" onClick={() => setActiveButton(3)}>Check Image</button>
    </div>
  );
};

export default SideButtons;