import React from 'react';
import './Home.css';
import profilePhoto from '../assets/profile.jpg';
import awsIcon from '../assets/amazonwebservices-original-wordmark.svg';

import jpLogo from '../assets/jpmorgan.png';
import metaLogo from '../assets/metaLogo.png';
import playPhoneLogo from '../assets/playphoneLogo.png';

function Home() {
  // ... existing skills object ...
  const skills = {
    frontend: [
      { name: "React", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" },
      { name: "TypeScript", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/typescript/typescript-original.svg" },
      { name: "JavaScript", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" },
      { name: "HTML5", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" },
      { name: "CSS3", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" }
    ],
    backend: [
      { name: "Java", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/java/java-original.svg" },
      { name: "Spring", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/spring/spring-original.svg" },
      { name: "Python", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" },
      { name: "PHP", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/php/php-original.svg" },
      { name: "C++", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cplusplus/cplusplus-original.svg" }
    ],
    tools: [
      { name: "Aws", icon: awsIcon },
      { name: "Athena", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/athena/athena-original.svg" },
      { name: "Glue", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/glue/glue-original.svg" },
      { name: "GraphQL", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/graphql/graphql-plain.svg" },
      { name: "Redux", icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redux/redux-original.svg" }
    ]
  };
  const experiences = [
    {
      company: "JPMorgan Chase & Co",
      position: "Associate Software Engineer",
      period: "May 2023 – Present",
      logo: jpLogo,
      description: [
        "Build a new digital retail bank for the UK market under the International Consumer Banking organization",
        "Leverage Java Spring technologies to develop and maintain core banking functionalities, ensuring compliance with industry standards and regulations",
        "Integrated AWS S3 for secure and efficient document storage, improving data accessibility and management",
        "Utilized Apache Kafka for robust event handling and messaging, facilitating real-time data processing and communication between microservices"
      ],
      technologies: ["Java", "Spring", "AWS S3", "Apache Kafka", "Microservices"]
    },
    {
      company: "Meta, Inc",
      position: "Software Engineering Intern",
      period: "May 2022 – Aug 2022",
      logo: metaLogo,
      description: [
        "Developed internal tools to facilitate monitoring, configuration, and maintenance of diverse components within Facebook's infrastructure",
        "Designed a reusable frontend user interface (UI) utilizing Typescript and React, and implemented robust backend controllers using PHP and GraphQL",
        "Conducted thorough unit testing to verify consistent behavior and functionality of the new UI and controllers"
      ],
      technologies: ["TypeScript", "React", "PHP", "GraphQL", "Unit Testing"]
    },
    {
      company: "Playphone, Inc",
      position: "Software Engineering Intern",
      period: "Sept 2018 – June 2019",
      logo: playPhoneLogo,
      description: [
        "Utilized Java Spring Boot to build a robust web server for the backend infrastructure",
        "Designed and optimized a SQL database that enables seamless scalability as the user base continues to grow",
        "Collaborated with previous and existing engineers to enhance the overall performance and efficiency of the platform through bug fixes and other optimization techniques"
      ],
      technologies: ["Java", "Spring Boot", "SQL", "Backend Development"]
    }
  ];

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
      <div className="home">
        <section id="home" className="hero-section">
          <div className="hero">
            <div className="profile-photo">
              <img src={profilePhoto} alt="Se Jin Lee" />
            </div>
            <h1>Hi, I'm Se Jin Lee</h1>
            <h2>Software Engineer</h2>
            <p>I build things for the web and love solving complex problems.</p>
            <div className="cta-buttons">
              <a href="#contact" className="cta-primary">Get in Touch</a>
              <a href="#projects" className="cta-secondary">View My Work</a>
            </div>
          </div>
          
  
        <section id="experience" className="experience-section">
          <h3>Work Experience</h3>
          <div className="timeline">
            {experiences.map((exp, index) => (
              <div className="experience-card" key={index}>
            <div className="experience-header">
              <div className="company-info">
                <img src={exp.logo} alt={exp.company} className="company-logo" />
                <div>
                  <h3>{exp.position}</h3>
                  <h4>{exp.company}</h4>
                  <span className="period">{exp.period}</span>
                </div>
              </div>
            </div>
            <ul className="responsibilities">
              {exp.description.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
            <div className="tech-used">
              {exp.technologies.map((tech, i) => (
                <span key={i} className="tech-badge">{tech}</span>
              ))}
            </div>
          </div>
            ))}
          </div>
        </section>
  
        <section id="projects" className="projects-section">
          
        <h3>Notable Projects</h3>
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
        </section>
  

        <div className="skills">
            <h3>Technical Skills</h3>
            <div className="skills-grid">
              <div className="skill-category">
                <h4>Frontend</h4>
                <div className="skill-icons">
                  {skills.frontend.map((skill, index) => (
                    <div key={index} className="skill-item">
                      <img src={skill.icon} alt={skill.name} />
                      <span>{skill.name}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="skill-category">
                <h4>Backend</h4>
                <div className="skill-icons">
                  {skills.backend.map((skill, index) => (
                    <div key={index} className="skill-item">
                      <img src={skill.icon} alt={skill.name} />
                      <span>{skill.name}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="skill-category">
                <h4>Tools & Technologies</h4>
                <div className="skill-icons">
                  {skills.tools.map((skill, index) => (
                    <div key={index} className="skill-item">
                      <img src={skill.icon} alt={skill.name} />
                      <span>{skill.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="contact" className="contact-section">
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

    </section>
      </div>
  );
}

export default Home;