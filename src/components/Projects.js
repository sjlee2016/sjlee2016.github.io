import React from 'react';
import './Projects.css';

function Projects() {
  const projects = [
    {
      title: "The Dialog System Technology Challenge",
      description: "Placed 1st place in track 4 of DSTC9, improving Facebook research team's situated interactive multi-modal conversational AI. Paper accepted at AAAI 2021 workshop.",
      technologies: ["Python", "Machine Learning", "LLM", "GPT-2"],
      github: "https://drive.google.com/file/d/1_-F_7mGcmmIm7S5IyH-puVZM5uelp7nu/view",
      achievements: [
        "Modified GPT-2 libraries to enable ensemble learning and improved response accuracy from 69% to 79%",
        "Co-authored paper: 'End-to-End Task-Oriented Multimodal Dialog System with GPT-2'"
      ],
      image: "https://via.placeholder.com/300x200"
    },
    {
      title: "Calligram",
      description: "Developed a deep neural network model for recognizing full-page Korean handwriting documents. Received Creative Award from HCI Korea 2021.",
      technologies: ["Python", "Computer Vision", "Django", "Deep Learning"],
      github: "https://github.com/sjlee2016/calligram",
      achievements: [
        "Achieved 98% text recognition accuracy for training data and 75% for testing data",
        "Co-authored paper: 'A Full-Page Hangul Handwriting Recognition Service Using OrigamiNet Models'"
      ],
      image: "https://via.placeholder.com/300x200"
    }
  ];

  return (
    <div className="projects-container">
      <h1>Notable Projects</h1>
      <div className="projects-grid">
        {projects.map((project, index) => (
          <div className="project-card" key={index}>
            <img src={project.image} alt={project.title} />
            <h3>{project.title}</h3>
            <p>{project.description}</p>
            <div className="achievements">
              {project.achievements.map((achievement, i) => (
                <li key={i}>{achievement}</li>
              ))}
            </div>
            <div className="tech-stack">
              {project.technologies.map((tech, i) => (
                <span key={i} className="tech-tag">{tech}</span>
              ))}
            </div>
            <div className="project-links">
              <a href={project.github} target="_blank" rel="noopener noreferrer">GitHub</a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Projects;