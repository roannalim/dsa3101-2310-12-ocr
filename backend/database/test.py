import mysql.connector
from PIL import Image
from io import BytesIO
import io
from datetime import datetime, timedelta
import pdb
##cronjob for deletion

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
def createImagesTable():
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))

        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE images (
            image_id int(10) NOT NULL auto_increment,
            image LONGBLOB,
            start_date DATE NOT NULL,
            expiry_date DATE NOT NULL,
            location VARCHAR(45) NOT NULL,
            PRIMARY KEY (image_id)
        );
        """

        cursor.execute(create_table_query)
        mysql.connection.commit()
        cursor.close()
        print("Table 'Images' created successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'
    # finally:
    #     if connection.is_connected():
    #         cursor.close()
    #         connection.close()
    #         print("MySQL connection is closed")

# Call the createimagesTable function to create the table
createImagesTable()

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
            name VARCHAR(45) NOT NULL,
            email VARCHAR(45) NOT NULL,
            PRIMARY KEY (user_id)
        );
        """
        cursor.execute(create_table_query)
        mysql.connection.commit()
        cursor.close()
        print("Table 'users' created successfully.")

    except Exception as e:
        return f'An error occurred: {str(e)}'
    

# Call the createUsersTable function to create the table
createUsersTable()


#to retrieve image
def retrieve_and_display_image(image_id):
    try:
        connection = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                             database=os.getenv("MYSQL_DB"),
                                             user=os.getenv("MYSQL_USER"),
                                             password=os.getenv("MYSQL_PASSWORD"))
        cursor = connection.cursor()
        
        # Retrieve the binary image data from the database
        query = "SELECT image FROM images WHERE image_id = %s"
        #filter by location and day , start_date = %s AND location= %s,, filter by user also, so user themselves can see their own hist
        #SELECT * --> to show all
        ##front end delete button, and func delete the photo from the db
        # tag username to the image, and add it to the users table , and tag it to an image
        #sql queries for filtering to the image table 
        cursor.execute(query, (image_id,))
        result = cursor.fetchone()

        if result:
            image_data = result[0]
            
            # Convert binary data to an image
            image = Image.open(BytesIO(image_data))
            
            # Display the image
            #image.show()
        else:
            print("Image with ID {} not found.".format(image_id))

    except mysql.connector.Error as error:
        print("Failed to retrieve and display the image: {}".format(error))


image_id_to_display = 1

# Call the retrieve_and_display_image function
retrieve_and_display_image(image_id_to_display)
UPLOAD_FOLDER='C:\\Users\\ACER\\Documents\\UNIY3S1\\DSA3101\\database\\uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def submit():
    db = mysql.connector.connect(host=os.getenv("MYSQL_HOST"),
                                 database=os.getenv("MYSQL_DB"),
                                 user=os.getenv("MYSQL_USER"),
                                 password=os.getenv("MYSQL_PASSWORD"))
    cursor = db.cursor()
    if request.method == 'POST':
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
        cursor.execute("INSERT INTO images (image_id, image ,start_date, expiry_date, location) VALUES (%s, %s, %s, %s, %s)",
                        (input['image_id'], image_data, start_date, expiry_date ,input['location']))
        db.commit()
        return "Successfully uploaded"
    try:
        return render_template("upload.html")
    except Exception as e:
        return f"Error: {e}"
    
if __name__ == 'main':
    app.run(debug=True)