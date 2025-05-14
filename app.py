import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def process_image(image_data):
    """
    Processes the image data to detect license plate shapes.

    Args:
        image_data (str): Base64 encoded image data.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected
              license plate and contains the cropped image and contour coordinates.
              Returns an empty list if no plates are found.  Returns None on error.
    """
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image_buffer = BytesIO(image_bytes)
        img = Image.open(image_buffer)
        img_np = np.array(img)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # 1. Preprocess the image for better contour detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 100, 200)  # Tune these thresholds if needed.

        # 2. Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Consider top 10

        plates = []
        for c in contours:
            # 3. Approximate the contour to a quadrilateral
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # Tune the 0.04 value.

            # 4. Check if it has 4 corners (potential license plate)
            if len(approx) == 4:
                # 5. Perspective Transform to get a rectangular plate shape
                (x, y, w, h) = cv2.boundingRect(approx)

                # Calculate the aspect ratio.  License plates have a specific aspect ratio
                aspect_ratio = w / float(h)
                # Typical license plate aspect ratio is between 1.5 and 6.  Adjust as needed.
                if 1.5 <= aspect_ratio <= 6:
                    # Reshape points for perspective transform
                    src_pts = approx.astype("float32")
                    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(img, M, (w, h))

                    # 6.  Further filtering (optional, but often helpful)
                    #  Check the area of the contour.  Too small is unlikely to be a plate.
                    area = cv2.contourArea(c)
                    if area > 100:  # Adjust this threshold as needed
                        # Convert the warped plate image to base64 for sending to the client
                        _, warped_buffer = cv2.imencode('.jpg', warped)
                        warped_base64 = base64.b64encode(warped_buffer).decode('utf-8')
                        plates.append({
                            'cropped_image': warped_base64,
                            'contour': approx.tolist(),  # Convert numpy array to list for JSON
                        })
        return plates
    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Explicitly return None on error

@app.route('/', methods=['GET'])
def index():
    """
    Renders the main page.
    """
    return render_template('index.html')

@app.route('/detect_license_plate', methods=['POST'])
def detect_license_plate():
    """
    Handles the image upload and license plate detection.
    """
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file:
            # Read the image data from the file
            image_data = file.read()
            # Encode the image data as base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Process the image to detect license plates
            plates = process_image(image_base64)

            if plates is None:
                return jsonify({'error': 'Error processing image'}), 500

            if not plates:
                return jsonify({'message': 'No license plate detected'}), 200
            else:
                return jsonify({'plates': plates}), 200

    except Exception as e:
        print(f"Exception in detect_license_plate: {e}")
        return jsonify({'error': f'Internal server error: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') #make accessible on the network
