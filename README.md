🛠️ How to Run Precision Vision on Your PC

Follow these steps to set up the environment and run the object detection system locally.

1. Prerequisites

Before starting, ensure you have the following installed:

Python (3.8 - 3.12): Download here

Git: Download here

2. Clone the Repository

Open your terminal (Command Prompt, PowerShell, or Bash) and run:

git clone [https://github.com/yourusername/precision-vision.git](https://github.com/yourusername/precision-vision.git)
cd precision-vision


3. Create a Virtual Environment (Recommended)

This keeps your project dependencies isolated from your main system.

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


4. Install Dependencies

Install the required AI and Web libraries using pip:

pip install ultralytics flask flask-cors pillow numpy


5. Add Your Model File

The system requires a YOLOv8 model file to function.

Locate your trained best.pt file.

Move it into the root directory of the project folder (/precision-vision/).

6. Run the Application

Start the Flask server:

python app.py


7. Access the Interface

Once the terminal shows Running on http://127.0.0.1:5000, follow these steps:

Open your web browser (Chrome, Edge, or Firefox).

Navigate to: http://127.0.0.1:5000

Use the Upload tab for static images or the Webcam tab for real-time detection.

⚠️ Common Troubleshooting

"ModuleNotFoundError": Ensure you activated the virtual environment in Step 3 before installing dependencies.

Webcam not starting: Make sure no other application (like Zoom or Teams) is using your camera.

Model Error: Ensure the file is named exactly best.pt and is in the same folder as app.py.
