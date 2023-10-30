from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
from datetime import datetime


app = Flask(__name__)
app.secret_key = "strongPassword"
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if check_credentials(username, password):
        session['user'] = username
        return redirect(url_for('main'))
    return redirect(url_for('login'))

# try not using the ==, check is there any library to authenticate.
def check_credentials(username, password):
    df = pd.read_csv('users.csv')
    matching_users = df[(df['username'] == username) & (df['password'] == password)]
    return not matching_users.empty

@app.route('/home')
def main():
    return render_template('Main.html')

@app.route('/image_processing', methods = ['POST'])
def image_processing():
    if 'image' in request.files:
        image = request.files['image']

        if image.filename != '':
            curr_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = curr_datetime + '.jpg'
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('image_preview', filename = filename))
        
    return "Image processing failed"

@app.route('/image_preview/<filename>')
def image_preview(filename):
    return render_template('Image_Preview.html', filename = filename)

@app.route('/edit_confirm', methods=['GET', 'POST'])
def edit_confirm():
    if request.method == 'POST':
        ## send data to backend
        return redirect(url_for('thank_you'))
    return render_template('Edit_Confirm.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/thank_you')
def thank_you():
    return render_template('Thank_you.html')

@app.route('/history')
def history():
    ## trigger function to display history data
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True)
