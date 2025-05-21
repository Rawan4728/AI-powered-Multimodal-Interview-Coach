# AI-powered-Multimodal-Interview-Coach


An AI-powered web application designed to help users enhance their interview skills through personalized, real-time feedback on verbal and non-verbal communication.

---

## 📌 Project Goal

The AI Interview Coach assists users in improving their interview performance by analyzing:

- **Verbal Analysis:** Evaluates spoken responses via speech-to-text transcription.
- **Facial Expression Analysis:** Assesses emotions and expressions during answers.
- **Posture Analysis:** Monitors body language and posture.
- **CV-Based Question Generation:** Generates tailored interview questions based on the user’s resume.

By delivering comprehensive, real-time feedback, the app helps users build confidence and refine their interview skills.

---

## 🧠 Project Architecture

The system consists of modular components to ensure scalability and maintainability:

- **User Interface (Gradio):**  
  Handles user interactions like CV uploads, video recordings, and displays feedback.

- **CV Processing Module:**  
  Parses resumes and uses a Retrieval-Augmented Generation (RAG) system to create personalized interview questions.

- **Interview Response Analysis:**  
  - Speech-to-Text Transcription with OpenAI Whisper.  
  - Facial Expression Analysis using DeepFace.  
  - Posture Analysis through MediaPipe and OpenCV.

- **RAG + GPT Question Answering:**  
  Retrieves relevant CV data via a vector store and generates context-aware interview questions and evaluations with GPT models.

- **Feedback Generation Engine:**  
  Integrates outputs from all analysis modules to produce detailed feedback on verbal communication, facial expressions, body language, and answer quality.

- **Output Delivery:**  
  Displays comprehensive feedback and performance metrics on the Gradio interface, guiding users on improvement areas.

---

## ✅ Testing Methodology

The project employs rigorous testing to ensure reliability and effectiveness:

- **Unit Testing:**  
  Validated core modules including CV parsing, speech transcription, emotion detection, and posture analysis under various input scenarios.

- **Integration Testing:**  
  Verified seamless interaction between modules from CV upload through to feedback delivery.

- **Performance Testing:**  
  Measured system responsiveness to ensure real-time feedback.

- **RAG System Evaluation with LangSmith:**  
  Used LangSmith to trace, debug, and assess the RAG pipeline, improving question relevance, answer coherence, and retrieval accuracy.
