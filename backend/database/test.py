import mysql.connector
from PIL import Image
from io import BytesIO
import io
from datetime import datetime, timedelta
import pdb
##cronjob for deletion

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import mysql.connector
from dotenv import load_dotenv

import pandas as pd
import base64

load_dotenv()

app = Flask(__name__)
app.secret_key = "strongPassword"

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

def is_authenticated(username, password):
    return users.get(username) == password

# Create the database
def createDatabase():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD")
        )

        cursor = connection.cursor()
        database_name = os.getenv("MYSQL_DB")

        create_database_query = f"CREATE DATABASE IF NOT EXISTS {database_name};"

        cursor.execute(create_database_query)

        print(f"Database '{database_name}' created successfully.")

    except Exception as e:
        print(f'An error occurred: {str(e)}')

    finally:
        cursor.close()
        connection.close()

# Call the createDatabase function to create the database
createDatabase()

# Create the users table
def createUsersTable():
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

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
    
# Call the createUsersTable function to create the table
createUsersTable()

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

# Create the images table
def createImagesTable():
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS images (
            image_id int(10) NOT NULL auto_increment,
            image LONGBLOB NOT NULL,
            start_date DATE NOT NULL,
            expiry_date DATE NOT NULL,
            location VARCHAR(45) NOT NULL,
            username VARCHAR(45) NOT NULL,
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

# Call the createImagesTable function to create the table
createImagesTable()

# Delete the images table
def deleteImagesTable():
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

        cursor = connection.cursor()
        create_table_query = """
        DROP TABLE IF EXISTS images;
        """

        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        print("Table 'images' deleted successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Call the deleteImagesTable function to delete the images table
# deleteImagesTable()

# Delete the users table
def deleteUsersTable():
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

        cursor = connection.cursor()
        create_table_query = """
        DROP TABLE IF EXISTS users;
        """

        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        print("Table 'users' deleted successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Call the deleteUsersTable function to delete the users table
# deleteUsersTable()

# Setting up the location to save the uploaded images
UPLOAD_FOLDER='/Users/richmondsin/Desktop/DSA3101/flask_mysql/uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

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

# Retrieve and display the image
def retrieve_and_display_image(image_id):
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))
        cursor = connection.cursor()
        
        # Retrieve the binary image data from the database
        query = "SELECT image FROM images WHERE image_id = %s"

        cursor.execute(query, (image_id,))
        result = cursor.fetchone()

        if result:
            image_data = result[0]
            
            # Convert binary data to an image
            image = Image.open(BytesIO(image_data))
            
            # Display the image
            image.show()
        else:
            print("Image with ID {} not found.".format(image_id))
        
    except mysql.connector.Error as error:
        print("Failed to retrieve and display the image: {}".format(error))

# Manual retrieval of image
# image_id = 1
# retrieve_and_display_image(image_id)

#1. filter by location and day , start_date = %s AND location= %s,, filter by user also, so user themselves can see their own hist
#2. front end delete button, and func delete the photo from the db
#3. tag username to the image, and add it to the users table , and tag it to an image

# Page to upload the images
@app.route('/upload', methods=['GET', 'POST'])
def submit():
    connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))
    cursor = connection.cursor()
    if request.method == 'POST':
        if 'username' in session:
            input = request.form
            f = request.files['image']
            image_data = f.read()
            photo_n = secure_filename(f.filename)

            # Calculate the start_date as the current date
            start_date = datetime.now().date()

            # Calculate the expiry_date as 3 months from the start_date
            expiry_date = start_date + timedelta(days=90)

            #Save the image to a folder
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_n))

            # Insert data into the MySQL database
            cursor.execute("INSERT INTO images (image, start_date, expiry_date, location, username) VALUES (%s, %s, %s, %s, %s)",
                            (image_data, start_date, expiry_date, input['location'], session['username']))
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
        file_name = session['file_name']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
             
        if request.method == 'POST':
            if request.form.get('confirm') == 'yes':
                # User confirmed the upload, do nothing
                return redirect(url_for('success'))
            else:
                connection = mysql.connector.connect(
                    host=os.getenv("MYSQL_HOST"),
                    database=os.getenv("MYSQL_DB"),
                    user=os.getenv("MYSQL_USER"),
                    password=os.getenv("MYSQL_PASSWORD")
                )
                cursor = connection.cursor()

                # Delete the record with the specified ID
                cursor.execute("DELETE FROM images WHERE image_id = %s", (image_id,))
                connection.commit()
                cursor.close()
                connection.close()

                # Delete the uploaded file from the file system
                os.remove(file_path)

                return redirect(url_for('cancelled'))
        return render_template("confirm_upload.html")

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
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))
        cursor = connection.cursor()
        
        # Retrieve image and associated information from the database
        query = "SELECT image_id, image, start_date, expiry_date, location, username FROM images"

        cursor.execute(query)
        results = cursor.fetchall()

        image_info = []

        if results:
            for result in results:
                image_id, image_data, start_date, expiry_date, location, username = result
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image_info.append({
                        'image_id': image_id,
                        'image': image,
                        'start_date': start_date,
                        'expiry_date': expiry_date,
                        'location': location,
                        'username': username
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
            connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                                 database=os.getenv("MYSQL_DB"),
                                                 user=os.getenv("MYSQL_USER"),
                                                 password=os.getenv("MYSQL_PASSWORD"))
            cursor = connection.cursor()
            
            # Retrieve image and associated information from the database
            query = "SELECT image_id, image, start_date, expiry_date, location, username FROM images WHERE image_id = %s"
            cursor.execute(query, (image_id,))
            result = cursor.fetchone()

            if result:
                image_id, image_data, start_date, expiry_date, location, username = result
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image_info = {
                        'image_id': image_id,
                        'image': image,
                        'start_date': start_date,
                        'expiry_date': expiry_date,
                        'location': location,
                        'username': username
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
            username = request.form.get('username')
            
            try:
                connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                                     database=os.getenv("MYSQL_DB"),
                                                     user=os.getenv("MYSQL_USER"),
                                                     password=os.getenv("MYSQL_PASSWORD"))
                cursor = connection.cursor()
                
                # Update the image details in the database
                query = "UPDATE images SET start_date = %s, expiry_date = %s, location = %s, username = %s WHERE image_id = %s"
                cursor.execute(query, (start_date, expiry_date, location, username, image_id))
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
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))
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

if __name__ == 'main':
    app.run(debug=True)

# flask --app test.py --debug run