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
- `test_mac.py`: This is the flask python file for the web app for mac users to test out the upload function only.
- `test_data.csv`: This is the 14 images data to be pre-inserted into the images table.
- `users.csv`: This is the 3 users data to be pre-inserted into users table.

### For mac users who cannot upload on docker but wants to test out the function: 
#### Prerequisites

Before running the test_mac Python script, make sure you have the following prerequisites in place:

1. **Python:** You need to have Python 3.10 installed on your system. You can download Python from the official website: [Python Downloads](https://www.python.org/downloads/).

2. **Dependencies:** This script relies on certain Python libraries and packages. You can install them by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

#### MySQL Server: 
You must have access to a MySQL server where the target database resides. If you need to create the MySQL server manually, please follow your organization's guidelines or the database provider's instructions.

#### Configuration
Before running the script, you need to make some specific configurations:

Environment Variables (.env)
In the root directory of the project, you should create a file named .env. This file should contain the following environment variables, specific to your database setup:

    
    DB_SERVER=your_server_name
    DB_NAME=your_database_name
    DB_USER=your_username
    DB_PASSWORD=your_password

#### Running the Script
To run the script, open your command line or terminal and navigate to the directory where the script is located. Then, execute the following command:

    
    flask --app test_mac.py --debug run