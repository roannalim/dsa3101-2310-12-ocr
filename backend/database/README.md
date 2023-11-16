# README

## General Information
This Python script is designed to perform specific tasks that involve interacting with a MySQL server, managing a database, and facilitating a flask application. Here, we will provide an overview of the script and its purpose.

### Database Information
This script connects to a MySQL server to manipulate data within a specific database. The details of the database, such as the server name, database name is included in the test.py folder. 

### Flask Application Information
This script includes a Flask application which serves the "What-A-Waste" website, providing an interface for users to upload, analyse, and modify the data. It's designed to interact with the MySQL server, handling image files and related tabular data (such as the image_id, start_date, expiry_date, location, username, gross_weight).

#### Key Features
- **Image Upload**: Users can upload truck dashboard images through a web form. The uploaded image is processed by an OCR model, and it is stored in the MySQL database with its related tabular data upon submission.
- **Dashboard**: Users can visit an interactive dashboard, developed through the R Shiny package, to look at the data analysis and gain insights about the data.
- **View History**: Images and related tabular data stored in the MySQL database are displayed to users in this page, enabling users to edit or delete data of past entries if necessary. There is a function to filter images according to start_date to expiry_date range, location and username. 

#### Related files
- `./static`: Folder containing logo, icons, and background images used in web interface.
- `./templates`: Folder containing web pages served by the Flask application, written in HTML and CSS.
- `./translations`: Folder containing essential files for the implementation of language options in the web interface.
- `./images`: Folder containing all 14 images to be pre-inserted into the MySQL database upon docker run.
- `database_dockerfile`: This is the dockerfile to build the database container.
- `flask_dockerfile`: This is the dockerfile to build the web container.
- `ocr_db.sql`: This is the SQL file to initialise the database. 
- `pytorch_ocr_model_function.py`: This is the file that contains the PyTorch EasyOCR processing function. 
- `requirements.txt`: This is the requirements file with the list of packages required.
- `test.py`: This is the flask python file for the web app.
- `test_data.csv`: This is the 14 images data to be pre-inserted into the images table.
- `users.csv`: This is the 3 users data to be pre-inserted into users table.