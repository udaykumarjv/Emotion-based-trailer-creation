# Emotion-based-trailer-creation
Human Assisted Trailer creation via task composition
Human Assisted Video Summary via Task Composition is an AI-powered movie trailer generation system that automatically creates emotionally engaging and character-centric trailers from full-length movies. The project combines advanced deep learning, computer vision, and video processing techniques to analyze movie scenes, detect characters, recognize emotions, and intelligently select impactful moments for trailer generation.

Traditional trailer creation requires manual editing, scene selection, and emotional storytelling by professional editors, which is time-consuming and subjective. This project automates the entire workflow using artificial intelligence, significantly reducing manual effort while maintaining high-quality cinematic output.

The system integrates:

MTCNN for multi-face detection
FaceNet for character recognition
VGG16-based CNN for facial emotion recognition
OpenCV & FFmpeg for video processing and scene extraction
Flask for backend services
SQLite3 for metadata storage
HTML, CSS, JavaScript for frontend development

The generated trailers focus on emotionally intense scenes and important characters, producing dynamic and meaningful summaries automatically.

🚀 Key Features
🎬 Automatic movie trailer generation
😀 Emotion-based scene selection
🧠 Deep learning-based face recognition
👥 Multi-character tracking across scenes
📹 Automatic scene segmentation
⚡ High-impact clip extraction
🌐 Web-based user interface
💾 SQLite database support
📥 Trailer preview and download support
🔄 Scalable modular architecture
🧠 Technologies Used
Backend
Python
Flask
Frontend
HTML5
CSS3
JavaScript
Bootstrap
AI & Deep Learning
MTCNN (Face Detection)
FaceNet (Face Recognition)
VGG16 CNN (Emotion Recognition)
TensorFlow / Keras
OpenCV
dlib
scikit-learn
Video Processing
OpenCV
FFmpeg
Database
SQLite3
Development Tools
VS Code / PyCharm
Jupyter Notebook
Git
📂 System Architecture

The system follows a layered architecture:

User Interface Layer
Allows users to upload movies, preview generated trailers, and download outputs.
Flask Backend Layer
Handles routing, processing workflows, and communication between modules.
AI Processing Layer
Performs:
Face Detection
Character Recognition
Emotion Classification
Video Processing Layer
Extracts frames, segments scenes, selects clips, and merges trailers.
Database Layer
Stores:
User data
Emotion labels
Scene timestamps
Character information
Trailer metadata
⚙️ Working Process
Step 1: Video Upload

The user uploads a full-length movie through the web interface.

Step 2: Frame Extraction

OpenCV extracts frames at regular intervals for efficient processing.

Step 3: Face Detection

MTCNN identifies all faces present in video frames.

Step 4: Character Recognition

FaceNet converts detected faces into embeddings and identifies characters.

Step 5: Emotion Detection

The VGG16 CNN classifies emotions such as:

Happy
Sad
Angry
Fear
Surprise
Neutral
Step 6: Scene Segmentation

Scenes are segmented based on:

Emotional intensity
Character importance
Temporal continuity
Step 7: Trailer Generation

High-impact clips are extracted and merged using FFmpeg.

Step 8: Final Output

The generated trailer is displayed and made available for download.

🎯 Objectives
Automate movie trailer generation
Reduce manual editing effort
Improve scalability in video summarization
Detect emotionally significant scenes
Generate character-focused trailers
Provide intelligent AI-driven media summarization
📊 Performance Results
Performance Metric	Result
Face Detection Accuracy	92.6%
Face Recognition Accuracy	88.4%
Emotion Recognition Accuracy	84.7%
Scene Selection Accuracy	89.2%
Trailer Duration	~2.5 Minutes
User Satisfaction	8.7/10
🧪 Testing

The project includes:

Unit Testing
Integration Testing
System Testing
Validation Testing
Performance Testing
Security Testing
Usability Testing

The system demonstrated stable performance with no major crashes during testing.

💡 Advantages
Fully automated trailer generation
Emotion-driven storytelling
Character-focused summarization
High scalability
Reduced human bias
Consistent trailer quality
Efficient processing using GPU acceleration
⚠️ Limitations
Reduced accuracy in low-light scenes
Facial occlusion affects recognition
High computational requirements
Limited cinematic transition effects
No audio sentiment analysis
🔮 Future Enhancements
Background music recommendation
Subtitle generation
Audio sentiment analysis
Advanced cinematic transitions
Personalized trailer themes
Real-time trailer generation
Cloud deployment support
📈 Applications
Movie trailer generation
OTT platform summarization
Content recommendation systems
AI-based video editing
Media analytics
Film production automation
🏁 Conclusion

This project successfully demonstrates an intelligent AI-driven approach for automated movie trailer generation using face recognition and emotion analysis. By combining computer vision, deep learning, and video processing techniques, the system produces emotionally engaging and character-centric trailers without manual intervention.

The proposed system significantly improves:

Speed
Scalability
Consistency
Personalization

compared to traditional manual editing methods, making it a powerful solution for modern media summarization and automated content generation.

📚 References
MTCNN Face Detection
FaceNet Face Recognition
VGG16 CNN Emotion Recognition
OpenCV Documentation
TensorFlow/Keras Documentation
FFmpeg Documentation
Flask Documentation
👨‍💻 Developed Using
Python
Flask
OpenCV
TensorFlow/Keras
FaceNet
MTCNN
SQLite3
HTML/CSS/JavaScript

                                                     Result 1
<img width="720" height="1017" alt="image" src="https://github.com/user-attachments/assets/947695ba-eaeb-4c01-8839-3942dae09272" />

                                                     Result 2
<img width="720" height="1017" alt="image" src="https://github.com/user-attachments/assets/97d44a1f-015c-4833-b9ca-53fc2d6a030a" />

                                                      Result 3
<img width="720" height="1017" alt="image" src="https://github.com/user-attachments/assets/b9f14261-a7cc-4756-b344-0e6fcbed7f91" />





