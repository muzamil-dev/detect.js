# DETECT: Deception Tracking Through Eye Cues Technology

## Overview

DETECT (Deception Tracking Through Eye Cues Technology) is a real-time, web-based system for analyzing potential deceptive behavior using eye-tracking and behavioral signals. It provides a non-invasive alternative to traditional polygraph-style methods by leveraging computer vision and deterministic signal processing techniques.

The system captures gaze patterns through a webcam or uploaded video, processes eye movement data in real time, and generates a deception likelihood score based on behavioral anomalies.

---

## Key Features

* **Real-Time Gaze Analysis**: Processes eye movement data live for immediate feedback.
* **Non-Invasive Eye Tracking**: Uses MediaPipe FaceMesh for facial landmark and iris detection.
* **Deterministic Detection Engine**: Computes deception indicators using gaze dispersion (variance) and directional acceleration.
* **Personalized Scoring**: Min-Max normalization adapts scoring across sessions for user-specific baselines.
* **Robust Signal Processing**: Applies Savitzky-Golay filtering and affine transformations for stability under head movement.
* **Web-Based Interface**: Supports live webcam input and video uploads.
* **Visualization Layer**: Displays real-time probability graphs and animated gaze tracking.

---

## Technology Stack

| Layer         | Technology                        | Description                                                     |
| ------------- | --------------------------------- | --------------------------------------------------------------- |
| Frontend      | Astro                             | High-performance web framework with minimal JavaScript overhead |
| Eye Tracking  | MediaPipe FaceMesh (MediaPipe.js) | Real-time facial landmark and iris tracking                     |
| Backend       | Go (Gochi Package)                | Low-latency, concurrent processing engine                       |
| Database      | PostgreSQL                        | Stores gaze metrics, timestamps, and session data               |
| Runtime       | Bun                               | Fast JavaScript runtime for frontend tooling                    |
| Communication | WebSockets                        | Real-time streaming between frontend and backend                |
| Deployment    | Docker + DigitalOcean VPS         | Containerized deployment environment                            |

---

## System Architecture

```mermaid
flowchart TD
    A[Video Input\n(Webcam / Upload)] --> B[Frontend: Astro]
    B --> C[MediaPipe FaceMesh\nEye Tracking]
    C --> D[Raw Gaze Coordinates]
    D --> E[WebSocket Stream]
    E --> F[Backend: Go Processing Engine]
    F --> G[Deterministic Analysis\n- Gaze Dispersion\n- Directional Acceleration]
    G --> H[Deception Score Calculation]
    H --> I[PostgreSQL Storage]
    H --> J[Real-Time Visualization]
    J --> K[User Dashboard]
```

---

## Data Flow

1. Video input is captured via webcam or uploaded file.
2. Astro frontend processes frames using MediaPipe FaceMesh.
3. Eye landmarks are converted into gaze coordinate data.
4. Data is streamed to backend via WebSockets.
5. Go backend applies filtering and statistical analysis.
6. Deception score is computed using deterministic rules.
7. Results are stored and visualized in real time.

---

## Installation & Deployment

The system is designed for containerized deployment using Docker.

### Core Tools

* **Docker**: Containerized deployment
* **Bun**: JavaScript runtime for frontend tooling
* **Conda**: Environment management for ML/MediaPipe components
* **GitHub / JIRA**: Version control and Agile workflow tracking

### Deployment Environment

* Hosted on DigitalOcean VPS
* Microservice-based Docker architecture
* WebSocket-based real-time communication layer

---

## Team

* Adam Bawatneh — Project Manager
* Brian Conn — Scrum Master & Frontend Developer
* Jeffrey Chang — Frontend Developer
* Muzamil Shamsi — Database & Frontend Developer
* Soham Ganguly — ML Engineer & Lead Full Stack Developer
* John Economou — Research & Backend Developer

---

## Project Status

The system is currently a functionally complete and deployable solution developed using Agile methodology.

### Epic Status

| Epic                                | Status      |
| ----------------------------------- | ----------- |
| Project Management and Coordination | Done        |
| Research and Data Collection        | Done        |
| Eye Tracking Implementation         | Done        |
| Anomaly Detection Algorithm         | In Progress |
| Web Application Development         | In Progress |
| Testing and Refinement              | To Do       |
| IRB Approval and Human Studies      | To Do       |

---

## Future Work

* Integration of advanced eye cues (pupil dilation, blink rate)
* Addition of speech and audio behavioral analysis
* IRB-approved validation studies with human participants
* Improved ML hybridization with statistical detection engine
* Enhanced personalization through long-term behavioral modeling

---

## License

To be defined by project stakeholders.
