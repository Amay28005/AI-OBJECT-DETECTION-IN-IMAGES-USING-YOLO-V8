from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from PIL import Image
import io
from ultralytics import YOLO

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow browser requests
CORS(app)

# --- YOUR MODEL INTEGRATION STARTS HERE ---

# 1. LOAD YOUR TRAINED YOLOv8 MODEL
# IMPORTANT: Replace 'yolov8n.pt' with the path to your own trained model file, e.g., 'best.pt'
# The model will be loaded once when the server starts.
try:
    print("--- Loading YOLOv8 model... ---")
    # This is the line you will change to load your model
    model = YOLO('best.pt') 
    print("--- YOLOv8 model loaded successfully! ---")
except Exception as e:
    print(f"--- ERROR loading model: {e} ---")
    print("--- Please make sure the model file 'yolov8n.pt' or your custom model is in the correct directory. ---")
    model = None

# --- MODEL PROCESSING FUNCTION ---
def run_model_on_image(image):
    """
    Processes an image with the loaded YOLOv8 model and returns predictions.
    """
    if model is None:
        return [{"error": "Model not loaded"}]
        
    # Run inference on the image
    results = model.predict(source=image, conf=0.25)
    
    # Extract bounding boxes, classes, and scores
    predictions = results[0].boxes
    class_names = results[0].names
    
    formatted_predictions = []
    for box in predictions:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        confidence = float(box.conf[0])
        
        # Get bounding box coordinates in (x_min, y_min, x_max, y_max) format
        bbox_xyxy = box.xyxy[0].cpu().numpy()
        x_min, y_min, x_max, y_max = bbox_xyxy
        
        # Calculate width and height for the required [x, y, width, height] format
        width = x_max - x_min
        height = y_max - y_min

        formatted_predictions.append({
            'bbox': [int(x_min), int(y_min), int(width), int(height)],
            'class': class_name,
            'score': confidence
        })
    
    print(f"Detected {len(formatted_predictions)} objects.")
    return formatted_predictions

# --- API AND WEB ROUTES ---

@app.route('/')
def home():
    """
    This function is called when a user goes to the main page.
    It serves the main HTML page for the web application.
    """
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    This is the API endpoint. The frontend sends images here.
    It receives an image, processes it with the model, and returns the results.
    """
    if request.method != 'POST':
        return jsonify({"error": "POST method required."}), 405

    try:
        # Get the image data from the request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "Missing 'image' key in request."}), 400
            
        # Decode the base64 image
        image_data = data['image']
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data))
        
        # Run the model
        predictions = run_model_on_image(image)
        
        # Return the results to the frontend
        return jsonify(predictions)

    except Exception as e:
        print(f"Error processing detection request: {e}")
        return jsonify({"error": "An error occurred during detection."}), 500

if __name__ == '__main__':
    # Starts the Flask server
    # Use 0.0.0.0 to make the app accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)

