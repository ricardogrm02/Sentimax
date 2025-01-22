import React from "react";
import ReactDOM from "react-dom";
import "./Header.css"; // Optional: Add styling for the header

function Header() {
  return ReactDOM.createPortal(
    <header className="sticky-header">
      <div className="header-content">
        <h1>Sentimax</h1>
      </div>
    </header>,
    document.getElementById("header-root") // Render to #header-root
  );
}

export default Header;