import mysql.connector
from PIL import Image
from io import BytesIO
import io
from datetime import datetime, timedelta
import pdb
from flask_babel import Babel #language translation

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import mysql.connector
from dotenv import load_dotenv

import pandas as pd
import base64

from pytorch_ocr_model_function import pytorch_easy_ocr

load_dotenv()

app = Flask(__name__)
app.secret_key = "strongPassword"
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = './translations'

babel = Babel(app)

@app.before_request
def before_request():
    # If the 'lang' parameter is specified in the URL, use it as the locale
    lang = request.args.get('lang')
    if lang:
        session['lang'] = lang
    elif 'lang' not in session:
        session['lang'] = app.config['BABEL_DEFAULT_LOCALE']

def get_locale():
    # If the 'lang' parameter is specified in the URL, use it as the locale
    return session.get('lang', app.config['BABEL_DEFAULT_LOCALE'])

babel.init_app(app, locale_selector=get_locale)

# Uncomment this for docker version
# def establish_sql_connection():
#     connection = mysql.connector.connect(
#     host="db",
#     user="root",
#     password="icebear123",
#     database="OCR_DB"
#     )
#     return connection

# Uncomment this for non-docker version
def establish_sql_connection():
    connection = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB"))
    return connection

# Uncomment this for non-docker version
# Create the users table
def createUsersTable():
    try:
        connection = establish_sql_connection()

        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            user_id int(10) NOT NULL auto_increment,
            username VARCHAR(45) NOT NULL UNIQUE,
            password VARCHAR(45) NOT NULL,
            PRIMARY KEY (user_id, username)
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        print("Table 'users' created successfully.")

    except Exception as e:
        print(f'An error occurred: {str(e)}')

# Uncomment this for non-docker version
# Call the createUsersTable function to create the table
createUsersTable()

# Uncomment this for non-docker version
# Create the images table
def createImagesTable():
    try:
        connection = establish_sql_connection()

        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS images (
            image_id int(10) NOT NULL auto_increment,
            image LONGBLOB NOT NULL,
            start_date DATE NOT NULL,
            expiry_date DATE NOT NULL,
            location VARCHAR(45) NOT NULL,
            username VARCHAR(45) NOT NULL,
            gross_weight int(10) NOT NULL,
            PRIMARY KEY (image_id)
        );
        """

        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        connection.close()
        print("Table 'images' created successfully.")

    except Exception as e:
        print(f'An error occurred: {str(e)}')

# Uncomment this for non-docker version
# Call the createImagesTable function to create the table
createImagesTable()

# Read csv file to store the user data
users_file = pd.read_csv('users.csv')

# Create a dictionary to store the username and password from the csv file
def storingUser(file):
    users = {}

    # for loop to read the username and password from the csv file
    for index, row in file.iterrows():
        username = row['username']
        password = row['password']

        # Create a dictionary to store the username and password
        users[username] = password
    return users  

# Call the storingUser function to store the username and password
users = storingUser(users_file)

# Read csv file to store the images data
images_file = pd.read_csv('test_data.csv')

# Create a dictionary to store the images data from the csv file
def storingImages(file):
    images = {}

    # for loop to read the username and password from the csv file
    for index, row in file.iterrows():
        # another line to read 10 images and store as image,, binary
            # Assuming you have a column in your CSV that contains file paths of the images
            image_path = f'./images/{index}.jpg'  # Adjust this to match your column name

            # Read the image file as binary data
            with open(image_path, 'rb') as f:
                binary_image_data = f.read() ##we nd this to replace images

            # Store the binary image data in the dictionary
            image = binary_image_data
            start_date = row['start_date'] 
            expiry_date = row['expiry_date']
            location=row['location']
            username=row['username']
            gross_weight=row['gross_weight']

            # Create a dictionary to store the images data         
            images[index] = image, start_date, expiry_date, location, username, gross_weight
    return images  

# Call the storingUser function to store the images data
images = storingImages(images_file)

def is_authenticated(username, password):
    return users.get(username) == password

# Insert data into the users table
def insertUsersTable(users):
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

        cursor = connection.cursor()
        insert_query = """
        INSERT INTO users (username, password) VALUES (%s, %s);
        """
        # for loop to execute the insert query from users
        for username, password in users.items():
            cursor.execute(insert_query, (username, password))
            
        connection.commit()
        cursor.close()
        print("Data inserted successfully.")

    except Exception as e:
        print(f'An error occurred: {str(e)}')

# Call the insertUsersTable function to insert data into the table
insertUsersTable(users)

# Insert data into the images table
def insertImagesTable(images):
    try:
        connection = establish_sql_connection()
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO images (image, start_date, expiry_date, location, username, gross_weight) VALUES (%s, %s, %s, %s, %s, %s);
        """
        # for loop to execute the insert query from users
        for image_id, (image, start_date, expiry_date, location, username, gross_weight) in images.items():
            cursor.execute(insert_query, (image, start_date, expiry_date, location, username, gross_weight))
            
        connection.commit()
        cursor.close()
        print("Data inserted successfully.")

    except Exception as e:
        print(f'An error occurred: {str(e)}')

# Call the insertImagesTable function to insert data into the table
insertImagesTable(images)

# Login page
@app.route('/', methods=['GET'])
def login_page():
    return render_template('index.html')

# Login page
@app.route('/', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Authenticate the user (e.g., check username and password)
    if is_authenticated(username, password):
        session['username'] = username
        return redirect(url_for('home'))
    else:
        return "Login failed"
    
# Home page
@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")

# Page to upload the images
@app.route('/upload', methods=['GET', 'POST'])
def submit():
    connection = establish_sql_connection()
    cursor = connection.cursor()
    if request.method == 'POST':
        if 'username' in session:
            input = request.form
            f = request.files['image']
            image_data = f.read()
            photo_n = secure_filename(f.filename)

            # OCR Processing
            ocr_result_str = pytorch_easy_ocr(image_data)
            # Attempting to convert the detected text to an integer
            try:
                gross_weight = int(ocr_result_str)
                print("Successfully converted to an integer:", gross_weight)
            except ValueError:
                print("Error: Could not convert detected text to an integer")
                gross_weight = 0

            # Calculate the start_date as the current date
            start_date = datetime.now().date()

            # Calculate the expiry_date as 3 months from the start_date
            expiry_date = start_date + timedelta(days=90)

            # Insert data into the MySQL database
            cursor.execute("INSERT INTO images (image, start_date, expiry_date, location, username, gross_weight) VALUES (%s, %s, %s, %s, %s, %s)",
                            (image_data, start_date, expiry_date, input['location'], session['username'], gross_weight))
            connection.commit()

            # Retrieve the auto-incremented ID of the inserted record
            inserted_id = cursor.lastrowid
            session['image_id'] = inserted_id
            session['file_name'] = photo_n
            cursor.close()
            connection.close()
            return redirect(url_for('confirm_upload'))
        else:
            return "User not authenticated"
    try:
        return render_template("upload.html")
    except Exception as e:
        return f"Error: {e}"

# Confirm the upload
@app.route('/confirm_upload', methods=['GET', 'POST'])
def confirm_upload():
    if 'image_id' in session:
        # Retrieve image_id and file_name from session
        image_id = session['image_id']

        image_data_base64 = None  # Initialize image_data as None
        image_info = None  # Initialize image_info as None

        if request.method == 'GET':
            # Fetch the image data from the database based on image_id
            connection = establish_sql_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT image, start_date, expiry_date, location, username, gross_weight FROM images WHERE image_id = %s", (image_id,))
            result = cursor.fetchone()
            if result:
                image_data_base64 = base64.b64encode(result[0]).decode('utf-8')
                image_info = {
                    'start_date': result[1],
                    'expiry_date': result[2],
                    'location': result[3],
                    'username': result[4],
                    'gross_weight': result[5]
                }
            else:
                return "Image not found in the database."
            cursor.close()
            connection.close()
                    
        if request.method == 'POST':
            if request.form.get('confirm') == 'yes':
                # Retrieve updated gross_weight from the form
                updated_gross_weight = request.form.get('updated_gross_weight')

                connection = establish_sql_connection()
                cursor = connection.cursor()

                # Update the gross_weight in the database
                cursor.execute("UPDATE images SET gross_weight = %s WHERE image_id = %s", (updated_gross_weight, image_id))
                connection.commit()
                cursor.close()
                connection.close()

                # User confirmed the upload, do nothing
                return redirect(url_for('success'))
            else:
                connection = establish_sql_connection()
                cursor = connection.cursor()

                # Delete the record with the specified ID
                cursor.execute("DELETE FROM images WHERE image_id = %s", (image_id,))
                connection.commit()
                cursor.close()
                connection.close()

                return redirect(url_for('cancelled'))
        return render_template("confirm_upload.html", image_data_base64=image_data_base64, image_info=image_info)

# Define the success route
@app.route('/success', methods=['GET'])
def success():
    return render_template("success.html")

# Define the cancelled route
@app.route('/cancelled', methods=['GET'])
def cancelled():
    return render_template("cancelled.html")

# Define a custom Jinja2 filter to convert the image to Base64
def image_to_base64(image):
    if image:
        image = image.convert('RGB') ##ADDED HERE CUS OF SOME ERROR
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        base64_data = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_data
    return ""

app.jinja_env.filters['to_base64'] = image_to_base64

# Retrieve and display the images
@app.route('/view_images', methods=['GET'])
def view_images():
    try:
        connection = establish_sql_connection()
        cursor = connection.cursor()
        
        # Retrieve image and associated information from the database
        query = "SELECT image_id, image, start_date, expiry_date, location, username, gross_weight FROM images"

        cursor.execute(query)
        results = cursor.fetchall()

        image_info = []

        if results:
            for result in results:
                image_id, image_data, start_date, expiry_date, location, username, gross_weight = result
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image_info.append({
                        'image_id': image_id,
                        'image': image,
                        'start_date': start_date,
                        'expiry_date': expiry_date,
                        'location': location,
                        'username': username,
                        'gross_weight':gross_weight
                    })
                else:
                    print("Empty image data found in a record. Skipping.")
            
            cursor.close()
            connection.close()
                        
            # Pass the list of image information to the template
            return render_template("view_images.html", image_info=image_info)
        else:
            print("No image found")
        
    except mysql.connector.Error as error:
        print("Failed to retrieve and display the images: {}".format(error))

# Edit data
@app.route('/edit_data', methods=['GET', 'POST'])
def edit_data():
    if request.method == 'GET':
        image_id = request.args.get('image_id')
        try:
            connection = establish_sql_connection()
            cursor = connection.cursor()
            
            # Retrieve image and associated information from the database
            query = "SELECT image_id, image, start_date, expiry_date, location, username, gross_weight FROM images WHERE image_id = %s"
            cursor.execute(query, (image_id,))
            result = cursor.fetchone()

            if result:
                image_id, image_data, start_date, expiry_date, location, username, gross_weight = result
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image_info = {
                        'image_id': image_id,
                        'image': image,
                        'start_date': start_date,
                        'expiry_date': expiry_date,
                        'location': location,
                        'username': username,
                        'gross_weight':gross_weight
                    }
                else:
                    print("Empty image data found for the selected image ID.")
            else:
                print("Image not found for the selected image ID.")

            cursor.close()
            connection.close()

            return render_template("edit_data.html", image_info=image_info)
        
        except mysql.connector.Error as error:
            print("Failed to retrieve the data for editing: {}".format(error))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update':
            image_id = request.form.get('image_id')
            start_date = request.form.get('start_date')
            expiry_date = request.form.get('expiry_date')
            location = request.form.get('location')
            username = session['username']
            gross_weight=request.form.get('gross_weight')
            
            try:
                connection = establish_sql_connection()
                cursor = connection.cursor()
                
                # Update the image details in the database
                cursor.execute("UPDATE images SET start_date = %s, expiry_date = %s, location = %s, username = %s, gross_weight=%s WHERE image_id = %s", (start_date, expiry_date, location, username, gross_weight, image_id))
                connection.commit()

                cursor.close()
                connection.close()

                # Redirect to the "View Images" page after updating
                return redirect('/view_images')
            
            except mysql.connector.Error as error:
                print("Failed to update the data: {}".format(error))
        
        # If the "Cancel" button is clicked, simply redirect to the "View Images" page
        return redirect('/view_images')

# Delete Image
@app.route('/delete_image', methods=['POST'])
def delete_image():
    image_id = request.form.get('image_id')
    try:
        connection = establish_sql_connection()
        cursor = connection.cursor()
        
        # Delete the image row from the database
        query = "DELETE FROM images WHERE image_id = %s"
        cursor.execute(query, (image_id,))
        connection.commit()

        cursor.close()
        connection.close()

        # Redirect to the "View Images" page after deleting
        return redirect('/view_images')
    
    except mysql.connector.Error as error:
        print("Failed to delete the image: {}".format(error))

#1. filter by location and day , start_date = %s AND location= %s,, filter by user also, so user themselves can see their own hist
@app.route('/filter_images', methods=['GET'])
def filter_images():
    loc = request.args.get('loc')
    st_date = request.args.get('st_date')
    end_date = request.args.get('end_date')
    user_id = request.args.get('user_id')

    # Handle defualt value
    if loc == "":
        loc = None
    if not loc:
        loc = None
    if not st_date:
        st_date = None
    if not end_date:
        end_date = None
    if not user_id:
        user_id = None

    try:
        connection = establish_sql_connection()
        cursor = connection.cursor()
        
        query = "SELECT image_id, image, start_date, expiry_date, location, username, gross_weight FROM images WHERE 1=1"

        params = []  # Create an empty list to store the parameters

        if st_date and end_date:
            query += " AND DATE(start_date) BETWEEN %s AND %s"
            params.extend([st_date, end_date])
        elif st_date:
            query += " AND DATE(start_date) >= %s"
            params.append(st_date)
        elif end_date:
            query += " AND DATE(start_date) <= %s"
            params.append(end_date)

        if loc:
            query += " AND location = %s"
            params.append(loc)

        if user_id:
            query += " AND username = %s"
            params.append(user_id)

        cursor.execute(query, params)  # Pass the query and parameters to cursor.execute

        results = cursor.fetchall()

        image_info = []

        if results:
            for result in results:
                image_id, image_data, start_date, expiry_date, location, username, gross_weight = result
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image_info.append({
                        'image_id': image_id,
                        'image': image,
                        'start_date': start_date,
                        'expiry_date': expiry_date,
                        'location': location,
                        'username': username,
                        'gross_weight':gross_weight
                    })
                else:
                    print("Empty image data found in a record. Skipping.")

        cursor.close()
        connection.close()

        return render_template("filtered_images.html", image_info=image_info, loc=loc, st_date=st_date, end_date=end_date, user_id=user_id)

    except mysql.connector.Error as error:
        print("Failed to retrieve and display the filtered images: {}".format(error))

if __name__ == 'main':
    app.run(debug=True)

# flask --app test.py --debug run

