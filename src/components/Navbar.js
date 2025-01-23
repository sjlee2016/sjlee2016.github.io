import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Se Jin Lee</Link>
      </div>
      <div className="navbar-links">

        <a href="/">Home</a>
        <a href="/projects">Projects</a>

        <a href="/experience">Experience</a>

        <a href="/contact">Contact</a>
      </div>
    </nav>
  );
}

export default Navbar;