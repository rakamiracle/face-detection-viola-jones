# 🔌 API Documentation - Face Detection System

## Base URL
```
http://localhost:5000
```

---

## 📍 Endpoints

### 1. GET `/`
**Description**: Serve the main application interface

**Response**: HTML page

**Example**:
```bash
curl http://localhost:5000/
```

---

### 2. POST `/detect`
**Description**: Process face detection on uploaded image

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "preprocess": "default",
  "detect_smile": true
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | - | Base64 encoded image with data URI prefix |
| `preprocess` | string | No | "default" | Preprocessing mode: `default`, `enhance`, `denoise`, `sharpen` |
| `detect_smile` | boolean | No | true | Enable/disable smile detection |

**Success Response** (200):
```json
{
  "success": true,
  "image": "base64_encoded_result_image",
  "face_count": 2,
  "eye_count": 4,
  "smile_count": 1,
  "process_time": 245,
  "resolution": "1920 x 1080",
  "saved_path": "results/result_20251204_135648.jpg",
  "details": [
    {
      "id": 1,
      "eyes": 2,
      "smile": true,
      "confidence": 100
    },
    {
      "id": 2,
      "eyes": 2,
      "smile": false,
      "confidence": 100
    }
  ]
}
```

**Error Response** (500):
```json
{
  "success": false,
  "message": "Error description"
}
```

**Example**:
```javascript
const response = await fetch('/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: imageBase64,
    preprocess: 'enhance',
    detect_smile: true
  })
});
const data = await response.json();
```

---

### 3. GET `/history`
**Description**: Retrieve detection history (latest 50 records)

**Response**:
```json
[
  {
    "id": 15,
    "timestamp": "2025-12-04 13:56:48",
    "face_count": 2,
    "eye_count": 4,
    "smile_count": 1,
    "process_time": 245,
    "image_path": "results/result_20251204_135648.jpg"
  },
  {
    "id": 14,
    "timestamp": "2025-12-04 13:45:22",
    "face_count": 1,
    "eye_count": 2,
    "smile_count": 0,
    "process_time": 189,
    "image_path": "results/result_20251204_134522.jpg"
  }
]
```

**Example**:
```bash
curl http://localhost:5000/history
```

```javascript
const response = await fetch('/history');
const history = await response.json();
```

---

### 4. GET `/stats`
**Description**: Get global detection statistics

**Response**:
```json
{
  "total_detections": 15,
  "total_faces": 28,
  "avg_process_time": 215.5
}
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `total_detections` | integer | Total number of detection operations performed |
| `total_faces` | integer | Total number of faces detected across all operations |
| `avg_process_time` | float | Average processing time in milliseconds |

**Example**:
```bash
curl http://localhost:5000/stats
```

```javascript
const response = await fetch('/stats');
const stats = await response.json();
```

---

## 🔧 Data Models

### Detection Result Object
```typescript
interface DetectionResult {
  success: boolean;
  image: string;              // Base64 encoded result image
  face_count: number;         // Total faces detected
  eye_count: number;          // Total eyes detected
  smile_count: number;        // Total smiles detected
  process_time: number;       // Processing time in ms
  resolution: string;         // Image resolution "width x height"
  saved_path: string;         // Path to saved result file
  details: FaceDetail[];      // Array of face details
}

interface FaceDetail {
  id: number;                 // Face ID (1-indexed)
  eyes: number;               // Number of eyes detected
  smile: boolean;             // Whether face is smiling
  confidence: number;         // Confidence score (0-100)
}
```

### History Item Object
```typescript
interface HistoryItem {
  id: number;                 // Database record ID
  timestamp: string;          // Detection timestamp "YYYY-MM-DD HH:MM:SS"
  face_count: number;         // Number of faces detected
  eye_count: number;          // Number of eyes detected
  smile_count: number;        // Number of smiles detected
  process_time: number;       // Processing time in ms
  image_path: string;         // Path to result image
}
```

### Statistics Object
```typescript
interface Statistics {
  total_detections: number;   // Total detection operations
  total_faces: number;        // Total faces detected
  avg_process_time: number;   // Average processing time in ms
}
```

---

## 📝 Usage Examples

### Complete Detection Workflow

```javascript
// 1. Upload and convert image to base64
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];
const reader = new FileReader();

reader.onload = async function(e) {
  const imageBase64 = e.target.result;
  
  // 2. Send detection request
  const response = await fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      preprocess: 'enhance',
      detect_smile: true
    })
  });
  
  const result = await response.json();
  
  if (result.success) {
    // 3. Display results
    console.log(`Detected ${result.face_count} faces`);
    console.log(`Processing time: ${result.process_time}ms`);
    
    // 4. Show annotated image
    const img = document.getElementById('resultImage');
    img.src = 'data:image/jpeg;base64,' + result.image;
    
    // 5. Display face details
    result.details.forEach(face => {
      console.log(`Face #${face.id}: ${face.eyes} eyes, ${face.smile ? 'smiling' : 'neutral'}`);
    });
    
    // 6. Refresh statistics
    const statsResponse = await fetch('/stats');
    const stats = await statsResponse.json();
    console.log(`Total detections: ${stats.total_detections}`);
  }
};

reader.readAsDataURL(file);
```

### Camera Capture and Detection

```javascript
// 1. Access camera
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const video = document.getElementById('video');
video.srcObject = stream;

// 2. Capture frame
function captureFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg');
}

// 3. Detect faces
const imageBase64 = captureFrame();
const response = await fetch('/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: imageBase64,
    preprocess: 'default',
    detect_smile: true
  })
});

const result = await response.json();
```

---

## ⚠️ Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| 500 Internal Server Error | Invalid image format | Ensure image is valid and properly encoded |
| 413 Payload Too Large | Image exceeds 16MB | Compress or resize image |
| 400 Bad Request | Missing required fields | Include all required parameters |
| Empty detection result | No faces in image | Try different preprocessing mode |

### Error Response Format
```json
{
  "success": false,
  "message": "Detailed error description"
}
```

---

## 🔒 Security Considerations

1. **File Size Limit**: Maximum 16MB per request
2. **Input Validation**: Backend validates image format
3. **CORS**: Not configured (localhost only)
4. **Rate Limiting**: Not implemented
5. **Authentication**: Not required

---

## 🚀 Performance Tips

1. **Image Size**: Smaller images process faster (recommended: 1920x1080 or less)
2. **Preprocessing**: Use only when necessary (adds processing time)
3. **Smile Detection**: Disable if not needed (reduces processing time by ~20%)
4. **Batch Processing**: Not supported (process one image at a time)

---

## 📊 Response Time Benchmarks

| Image Size | Faces | Preprocessing | Avg Time |
|------------|-------|---------------|----------|
| 640x480 | 1 | None | ~150ms |
| 1280x720 | 2 | None | ~200ms |
| 1920x1080 | 3 | None | ~300ms |
| 1920x1080 | 3 | Enhanced | ~450ms |
| 3840x2160 | 5 | None | ~800ms |

*Benchmarks may vary based on hardware*

---

## 🔄 API Versioning

**Current Version**: 1.0  
**Stability**: Stable  
**Breaking Changes**: None planned

---

## 📞 Support

For issues or questions:
- GitHub: [rakamiracle](https://github.com/rakamiracle)
- Project: face-detection-viola-jones
