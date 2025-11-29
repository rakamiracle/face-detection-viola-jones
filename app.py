"""
=================================================================
PROYEK DETEKSI WAJAH - VIOLA JONES (ADVANCED VERSION - FIXED)
=================================================================

INSTALL:
pip install flask opencv-python pillow numpy

JALANKAN:
python3 app.py
=================================================================
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import time
from datetime import datetime
import sqlite3

# INISIALISASI
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# BUAT FOLDER
for folder in ['uploads', 'results', 'database']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# LOAD MODEL VIOLA-JONES
print("🔄 Loading Viola-Jones models...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
print("✅ Models loaded!")

# DATABASE SETUP
def init_db():
    conn = sqlite3.connect('database/faces.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  face_count INTEGER,
                  eye_count INTEGER,
                  smile_count INTEGER,
                  process_time INTEGER,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()


# ============================================================
# FUNGSI PREPROCESSING
# ============================================================
def preprocessing(image, mode='default'):
    """
    Preprocessing gambar untuk hasil lebih baik
    """
    if mode == 'enhance':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    elif mode == 'denoise':
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    elif mode == 'sharpen':
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    return image


# ============================================================
# FUNGSI DETEKSI WAJAH ADVANCED
# ============================================================
def deteksi_wajah_advanced(gambar, detect_smile=True):
    """
    Deteksi wajah, mata, dan senyum dengan detail per wajah
    """
    mulai = time.time()
    hasil_detail = []
    
    # Convert ke grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    wajah_list = face_cascade.detectMultiScale(
        abu_abu,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    total_mata = 0
    total_senyum = 0
    
    # Proses setiap wajah
    for idx, (x, y, w, h) in enumerate(wajah_list, 1):
        # Kotak HIJAU untuk wajah
        cv2.rectangle(gambar, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(gambar, f'WAJAH #{idx}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ROI
        roi_abu = abu_abu[y:y+h, x:x+w]
        roi_warna = gambar[y:y+h, x:x+w]
        
        # Deteksi MATA
        mata_list = eye_cascade.detectMultiScale(roi_abu, 1.1, 10, minSize=(20, 20))
        jumlah_mata = len(mata_list)
        total_mata += jumlah_mata
        
        for (ex, ey, ew, eh) in mata_list:
            cv2.rectangle(roi_warna, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.circle(roi_warna, (ex+ew//2, ey+eh//2), 3, (0, 255, 255), -1)
        
        # Deteksi SENYUM
        ada_senyum = False
        if detect_smile:
            roi_mulut_abu = roi_abu[h//2:, :]
            roi_mulut_warna = roi_warna[h//2:, :]
            
            senyum_list = smile_cascade.detectMultiScale(
                roi_mulut_abu, 
                scaleFactor=1.8, 
                minNeighbors=20,
                minSize=(25, 25)
            )
            
            if len(senyum_list) > 0:
                ada_senyum = True
                total_senyum += 1
                cv2.putText(roi_warna, 'SMILE', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Hitung confidence
        confidence = min(jumlah_mata * 50, 100)
        
        # Info di samping wajah
        info_y = y + 20
        cv2.putText(gambar, f"Eyes: {jumlah_mata}", (x+w+10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(gambar, f"Conf: {confidence}%", (x+w+10, info_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Simpan detail (FIXED: convert semua ke Python native types)
        hasil_detail.append({
            'id': int(idx),
            'eyes': int(jumlah_mata),
            'smile': bool(ada_senyum),
            'confidence': int(confidence)
        })
    
    # Hitung waktu
    waktu_ms = int((time.time() - mulai) * 1000)
    
    # Gambar summary di atas
    cv2.rectangle(gambar, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.putText(gambar, f"Wajah: {len(wajah_list)}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(gambar, f"Mata: {total_mata}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(gambar, f"Senyum: {total_senyum}", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return gambar, len(wajah_list), total_mata, total_senyum, waktu_ms, hasil_detail


# ============================================================
# SIMPAN KE DATABASE
# ============================================================
def simpan_ke_db(face_count, eye_count, smile_count, process_time, image_path):
    conn = sqlite3.connect('database/faces.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO detections 
                 (timestamp, face_count, eye_count, smile_count, process_time, image_path)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (timestamp, face_count, eye_count, smile_count, process_time, image_path))
    conn.commit()
    conn.close()


# ============================================================
# ROUTE: HOMEPAGE
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# ROUTE: DETEKSI WAJAH (FIXED)
# ============================================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        gambar_base64 = data['image']
        preprocess_mode = data.get('preprocess', 'default')
        detect_smile = data.get('detect_smile', True)
        
        # Decode image
        gambar_base64 = gambar_base64.split(',')[1]
        gambar_bytes = base64.b64decode(gambar_base64)
        
        pil_image = Image.open(io.BytesIO(gambar_bytes))
        gambar_array = np.array(pil_image)
        
        if len(gambar_array.shape) == 3:
            gambar_array = cv2.cvtColor(gambar_array, cv2.COLOR_RGB2BGR)
        
        # Preprocessing
        if preprocess_mode != 'default':
            gambar_array = preprocessing(gambar_array, preprocess_mode)
        
        # Deteksi wajah
        hasil, jumlah_wajah, jumlah_mata, jumlah_senyum, waktu, detail = deteksi_wajah_advanced(
            gambar_array, detect_smile=detect_smile
        )
        
        # Simpan hasil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"results/result_{timestamp}.jpg"
        cv2.imwrite(result_path, hasil)
        
        # Simpan ke database
        simpan_ke_db(jumlah_wajah, jumlah_mata, jumlah_senyum, waktu, result_path)
        
        # Convert hasil ke base64
        _, buffer = cv2.imencode('.jpg', hasil)
        hasil_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # FIXED: detail sudah dalam format Python native types
        return jsonify({
            'success': True,
            'image': hasil_base64,
            'face_count': int(jumlah_wajah),
            'eye_count': int(jumlah_mata),
            'smile_count': int(jumlah_senyum),
            'process_time': int(waktu),
            'resolution': f"{gambar_array.shape[1]} x {gambar_array.shape[0]}",
            'details': detail,  # Sudah aman untuk JSON
            'saved_path': result_path
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================
# ROUTE: RIWAYAT DETEKSI
# ============================================================
@app.route('/history')
def history():
    try:
        conn = sqlite3.connect('database/faces.db')
        c = conn.cursor()
        c.execute('SELECT * FROM detections ORDER BY id DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        
        history_data = []
        for row in rows:
            history_data.append({
                'id': int(row[0]),
                'timestamp': str(row[1]),
                'face_count': int(row[2]),
                'eye_count': int(row[3]),
                'smile_count': int(row[4]),
                'process_time': int(row[5]),
                'image_path': str(row[6])
            })
        
        return jsonify(history_data)
    except Exception as e:
        return jsonify([])


# ============================================================
# ROUTE: STATISTIK
# ============================================================
@app.route('/stats')
def stats():
    try:
        conn = sqlite3.connect('database/faces.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*), SUM(face_count), AVG(process_time) FROM detections')
        row = c.fetchone()
        
        conn.close()
        
        return jsonify({
            'total_detections': int(row[0] or 0),
            'total_faces': int(row[1] or 0),
            'avg_process_time': round(float(row[2] or 0), 2)
        })
    except Exception as e:
        return jsonify({
            'total_detections': 0,
            'total_faces': 0,
            'avg_process_time': 0
        })


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("   🚀 SERVER DETEKSI WAJAH - VIOLA JONES (ADVANCED)")
    print("="*70)
    print("   📍 URL       : http://localhost:5000")
    print("   🔧 Backend   : Flask + OpenCV")
    print("   🎯 Algoritma : Viola-Jones Cascade Classifier")
    print("   ✨ Fitur     : Multi-face, Smile, History, Stats")
    print("="*70)
    print("\n   ⏳ Server running... (Ctrl+C untuk stop)\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)