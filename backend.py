from flask import Flask, jsonify, request
import requests
from flask_cors import CORS
from recognition_try import run_recognition

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get_image_url', methods=['POST'])
def get_image_url():
    # Get the image URL from the request payload
    image_url = request.json['image_url']
    run_recognition(image_url)
    return jsonify(status='success', message='Image processing complete',image_url=image_url)

if __name__ == '__main__':
    app.run()

