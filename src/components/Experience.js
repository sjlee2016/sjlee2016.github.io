import React from 'react';
import './Experience.css';
import jpLogo from '../assets/jpmorgan.png';
import metaLogo from '../assets/metaLogo.png';
import playPhoneLogo from '../assets/playphoneLogo.png';

function Experience() {
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

  return (
    <div className="experience-container">
      <h1>Work Experience</h1>
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
    </div>
  );
}

export default Experience;