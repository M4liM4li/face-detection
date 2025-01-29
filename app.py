import os
import face_recognition
from flask import Flask, request, jsonify
import io
import requests
from flask_cors import CORS
import logging
from typing import Tuple, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'NODE_JS_URL': 'https://check-name-server.vercel.app/api/attendance',
    'FACE_FOLDER': 'Face/',
    'TOLERANCE': 0.5,
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,  # seconds
    'MIN_CONFIDENCE': 65.0  # Added minimum confidence threshold
}
app = Flask(__name__)
CORS(app)

class FaceRecognitionError(Exception):
    """Custom exception for face recognition errors"""
    pass

def load_known_faces() -> Tuple[List[any], List[str]]:
    """
    Load and encode known faces from the faces directory
    Returns tuple of (face_encodings, face_names)
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(CONFIG['FACE_FOLDER']):
        logger.error(f"Face folder {CONFIG['FACE_FOLDER']} not found")
        return known_face_encodings, known_face_names

    for filename in os.listdir(CONFIG['FACE_FOLDER']):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(CONFIG['FACE_FOLDER'], filename)
            try:
                logger.info(f"Loading face: {filename}")
                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    logger.info(f"Successfully loaded face: {name}")
                else:
                    logger.warning(f"No face detected in: {filename}")
            except Exception as e:
                logger.error(f"Error loading face {filename}: {str(e)}")

    return known_face_encodings, known_face_names

def send_to_nodejs(data: dict, retries: int = CONFIG['MAX_RETRIES']) -> Optional[dict]:
    """
    Send data to Node.js server with retry mechanism
    Returns response from Node.js or None if failed
    """
    for attempt in range(retries):
        try:
            logger.info(f"Sending data to Node.js: {data}") 
            response = requests.post(
                CONFIG['NODE_JS_URL'],
                json=data,
                timeout=5  # 5 seconds timeout
            )
            if response.status_code == 200:
                return response.json()
            
            logger.warning(f"Attempt {attempt + 1}: Failed to send data to Node.js. Status: {response.status_code}")
            
            # Successfully sent data even if Node.js returns non-200
            return {
                'status': 'partial_success',
                'message': 'Data sent but Node.js returned non-200 status',
                'node_status': response.status_code
            }
            
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1}: Error sending data to Node.js: {str(e)}")
            if attempt < retries - 1:
                time.sleep(CONFIG['RETRY_DELAY'])
            
    return None

# Load faces on startup
known_face_encodings, known_face_names = load_known_faces()

@app.route('/compare-face', methods=['POST'])
def compare_faces():
    """Handle face comparison requests"""
    
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image uploaded'
        }), 400

    try:
        # Process uploaded image
        image_file = request.files['image']
        image_data = image_file.read()
        
        if not image_data:
            raise FaceRecognitionError("Empty image data")
        
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        face_encodings = face_recognition.face_encodings(image)
        
        image_file = request.files['image']
        print(f"File type: {image_file.content_type}")
        print(f"File size: {len(image_file.read())} bytes")
        if not face_encodings:
            return jsonify({
                'status': 'error',
                'message': 'No face detected in uploaded image'
            }), 400

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding,
                tolerance=CONFIG['TOLERANCE']
            )
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if True in matches:
                best_match_index = matches.index(True)
                matched_name = known_face_names[best_match_index]
                confidence = (1 - face_distances[best_match_index]) * 100

                print(f"Matched Name: {matched_name}, Confidence: {confidence:.2f}%")

                # Create response data
                result_data = {
                    'name': matched_name,
                    'confidence': f'{confidence:.2f}%'
                }

                response_data = {
                    'status': 'success',
                    'message': f'Match Found! Name: {matched_name}',
                    'name': matched_name,
                    'confidence': f'{confidence:.2f}%',
                }

                # Only send to Node.js if confidence meets threshold
                if confidence >= CONFIG['MIN_CONFIDENCE']:
                    nodejs_response = send_to_nodejs(result_data)
                    if nodejs_response:
                        response_data['node_response'] = nodejs_response
                    else:
                        response_data['node_status'] = 'Data sent but no response from Node.js'
                else:
                    response_data['message'] = f'Match found but confidence too low ({confidence:.2f}%)'
                    response_data['node_status'] = 'Data not sent due to low confidence'

                return jsonify(response_data)

        return jsonify({
            'status': 'not_found',
            'message': 'No Match Found',
            'name': None,
            'confidence': '0%'
        })

    except FaceRecognitionError as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Face recognition error: {str(e)}'
        }), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing the image: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)