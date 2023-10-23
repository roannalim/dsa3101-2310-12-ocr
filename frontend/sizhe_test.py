from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('Main.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/thank_you')
def thank_you():
    return render_template('Thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
