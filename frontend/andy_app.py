from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def main():
    return render_template('Upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)

    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(image_path)
        return redirect(url_for('image_preview', image_path=image_path))

@app.route('/image_preview/<path:image_path>', methods=['GET'])
def image_preview(image_path):
    return render_template('image_preview.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug = True)
