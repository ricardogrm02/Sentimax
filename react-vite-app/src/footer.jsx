// Footer.jsx

import React from "react";
import ReactDOM from "react-dom";
import "./Footer.css"; // Optional: Add styling for the footer

function Footer() {
  return ReactDOM.createPortal(
    <footer className="sticky-footer">
      <div className="footer-content">
        <p>&copy; 2025 Sentimax. All rights reserved.</p>
      </div>
    </footer>,
    document.getElementById("footer-root") // Render to #footer-root
  );
}

export default Footer;
