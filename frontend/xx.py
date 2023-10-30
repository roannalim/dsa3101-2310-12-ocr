from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/image_processing', methods=['POST'])
def image_processing():
    if 'image' in request.files:
        image = request.files['image']

        if image.filename != '':
            curr_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = curr_datetime + '.jpg'
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('image_preview', filename=filename))

    return "Image processing failed"

@app.route('/image_preview/<filename>', methods=['GET', 'POST'])
def image_preview(filename):
    if request.method == 'POST':
        return redirect(url_for('edit_confirm'))
    return render_template('Image_Preview.html', filename=filename)

@app.route('/edit_confirm', methods=['GET', 'POST'])
def edit_confirm():
    if request.method == 'POST':
        return redirect(url_for('thank_you'))
    return render_template('Edit_Confirm.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/thank_you')
def thank_you():
    return render_template('Thank_you.html')

import logging

if __name__ == '__main__':
    app.debug = True
    app.run()
    logging.basicConfig(filename='app.log', level=logging.DEBUG)