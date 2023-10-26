import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import pdb
from PIL import Image
from io import BytesIO
import io
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)

app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)

def createPicturesTable():
    try:
        cur = mysql.connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS pictures (
            image_id INT(10) NOT NULL auto_increment,
            image LONGBLOB,
            start_date DATE NOT NULL,
            expiry_date DATE NOT NULL,
            location VARCHAR(45) NOT NULL,
            PRIMARY KEY (image_id)
        );
        """
        cur.execute(create_table_query)
        mysql.connection.commit()
        cur.close()
        print("Table 'pictures' created successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Call the createPicturesTable function to create the table
createPicturesTable()

def createUsersTable():
    try:
        cur = mysql.connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            user_id int(10) NOT NULL auto_increment,
            name VARCHAR(45) NOT NULL,
            email VARCHAR(45) NOT NULL,
            PRIMARY KEY (id)
        );
        """
        cur.execute(create_table_query)
        mysql.connection.commit()
        cur.close()
        print("Table 'users' created successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Call the createUsersTable function to create the table
createUsersTable()

# Insert images into the 'pictures' table
def insertImage(image_path):
    try:
        cur = mysql.connection.cursor()

        with open(image_path, 'rb') as file:
            image_data = file.read()
            
        image = Image.open(io.BytesIO(image_data))
        image = image.rotate(90)

        # Convert the image back to binary data
        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            image_data = output.getvalue()
        
        # Calculate the start_date as the current date
        start_date = datetime.now().date()

        # Calculate the expiry_date as 3 months from the start_date
        expiry_date = start_date + timedelta(days=90)

        # location
        location = "UTown"

        # Insert the image data along with start_date and expiry_date
        insert_query = "INSERT INTO pictures (image, start_date, expiry_date) VALUES (%s, %s, %s, %s)"
        cur.execute(insert_query, (image_data, start_date, expiry_date, location))
        mysql.connection.commit()
        cur.close()
        print("Image inserted successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Call the insertImage function to insert an image into the 'pictures' table
insertImage("20231006_084417.jpg")

# Retrieve and display an image from the 'pictures' table
def retrieve_and_display_image(image_id):
    try:
        cur = mysql.connection.cursor()
        
        # Retrieve the binary image data from the database
        query = "SELECT image FROM pictures WHERE image_id = %s"
        cur.execute(query, (image_id,))
        result = cur.fetchone()

        if result:
            image_data = result[0]
            
            # Convert binary data to an image
            image = Image.open(BytesIO(image_data))
            
            # Display the image
            image.show()
        else:
            print("Image with ID {} not found.".format(image_id))

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Manually set the id to display
image_id_to_display = 1

# Call the retrieve_and_display_image function
retrieve_and_display_image(image_id_to_display)

@app.route('/home')
def home():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    fetchdata = cur.fetchall()
    cur.close()
    return render_template('home.html', data = fetchdata)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        # Fetch form data
        userDetails = request.form
        id = userDetails['id']
        name = userDetails['name']
        email = userDetails['email']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(user_id, name, email) VALUES(%s, %s, %s)",(id, name, email))
        mysql.connection.commit()
        cur.close()
        return 'success'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
