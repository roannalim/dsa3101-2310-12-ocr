import os
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_mysqldb import MySQL
import pdb

load_dotenv()

app = Flask(__name__)

app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)

@app.route('/')
def home():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM user")
    fetchdata = cur.fetchall()
    cur.close()
    return render_template('home.html', data = fetchdata)

if __name__ == '__main__':
    app.run(debug=True)
