import React from 'react';
import './Contact.css';

function Contact() {
  return (
    <div className="contact-container">
      <h1>Get In Touch</h1>
      <p className="contact-intro">
        I'm currently based in New York and open to new opportunities. Feel free to reach out!
      </p>
      
      <div className="contact-methods">
        <div className="contact-method">
          <h3>📍 Location</h3>
          <p>New York, NY</p>
        </div>

        <div className="contact-method">
          <h3>📧 Email</h3>
          <a href="mailto:sjleesogang@gmail.com">sjleesogang@gmail.com</a>
        </div>
        
        <div className="contact-method">
          <h3>💼 GitHub</h3>
          <a href="https://github.com/sjlee2016" target="_blank" rel="noopener noreferrer">
            github.com/sjlee2016
          </a>
        </div>

        <div className="contact-method">
          <h3>🎓 Education</h3>
          <p>MS in Computer Science</p>
          <p>New York University</p>
          <p>GPA: 3.727/4.0</p>
        </div>
      </div>
    </div>
  );
}

export default Contact;