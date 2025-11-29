# 🎯 Face Detection - Viola Jones Algorithm

Advanced face detection web application using Viola-Jones algorithm with Flask backend and modern UI.

![Face Detection Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)

## ✨ Features

- 👤 **Multi-Face Detection** - Detect multiple faces in one image
- 👁️ **Eye Detection** - Automatically detect eyes on each face
- 😊 **Smile Detection** - Identify smiling faces
- 🎨 **Image Preprocessing** - Multiple modes (enhance, denoise, sharpen)
- 📷 **Camera Support** - Real-time capture from webcam
- 📊 **Statistics & History** - Track all detections with detailed stats
- 💾 **Export Results** - Download processed images
- 🎨 **Modern UI** - Clean, responsive, and elegant interface

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV (Viola-Jones Cascade)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Algorithm**: Haar Cascade Classifier

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face-detection-viola-jones.git
cd face-detection-viola-jones
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install flask opencv-python pillow numpy
```

4. **Run the application**
```bash
python3 app.py
```

5. **Open in browser**
```
http://localhost:5000
```

## 📁 Project Structure
```
face-detection-viola-jones/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend UI
├── uploads/              # Uploaded images (auto-created)
├── results/              # Processed results (auto-created)
├── database/             # SQLite database (auto-created)
├── venv/                 # Virtual environment
├── .gitignore
└── README.md
```

## 🚀 Usage

1. **Upload Image** - Click "Upload Image" and select a photo
2. **Or Use Camera** - Click "Open Camera" to capture from webcam
3. **Configure Settings** - Choose preprocessing mode and enable/disable smile detection
4. **Detect** - Click "Start Detection" to process
5. **View Results** - See detected faces, eyes, and smiles with statistics
6. **Download** - Save the processed image with annotations

## 📊 Detection Statistics

The application tracks:
- Total number of detections
- Total faces detected
- Average processing time
- Complete history of all detections

## 🎨 Preprocessing Modes

- **Default** - Standard processing
- **Enhanced Contrast** - Improve visibility in low-light images
- **Noise Reduction** - Remove image noise
- **Sharpening** - Enhance edge details

## 🔬 Algorithm Details

This project uses the **Viola-Jones algorithm** with Haar Cascade Classifiers:

- **Face Detection**: `haarcascade_frontalface_default.xml`
- **Eye Detection**: `haarcascade_eye.xml`
- **Smile Detection**: `haarcascade_smile.xml`

## 📝 Requirements
```
Flask>=2.0.0
opencv-python>=4.5.0
Pillow>=9.0.0
numpy>=1.21.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV community for the Haar Cascade models
- Flask framework
- Viola-Jones algorithm creators

---

⭐ If you find this project useful, please give it a star!