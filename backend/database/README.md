# README

## General Information

This Python script is designed to perform specific tasks that involve interacting with a MySQL server, managing a database, and facilitating a flask application. Here, we will provide an overview of the script, its purpose, and the steps required to run it successfully.

### Database Information

This script connects to a MySQL server to manipulate data within a specific database. The details of the database, such as the server name, database name, and credentials, need to be configured in the `.env` file, which will be explained in the next section.

### Flask Applicaton Information

This script includes a Flask application which serves the "What-A-Waste" website, providing an interface for users to upload, retrieve, and edit data. It's designed to interact with the MySQL server, handling image files and related data (such as the gross weight, time of image upload etc).

#### Key Features

- **Image Upload**: Users can upload truck dashboard images through a web form. The uploaded image is processed by an OCR model, and it is stored in the MySQL database with its related data upon submission.
- **Dashboard**: Users can visit an interactive dashboard, developed through R Shiny package, to look at the data analysis and gain insights about the data.
- **View History**: Images and related data stored in the MySQL database are displayed to users in a web interface, enabling users to edit or delete data of past entries if necessary.

#### Related files

- `./static`: Folder containing logo, icons, and background images used in web interface.
- `./templats`: Folder containing web pages served by the Flask application, written in HTML and CSS.
- `./translations`: Folder containing essential files for the implementation of language option in the web interface.

## Prerequisites

Before running this Python script, make sure you have the following prerequisites in place:

1. **Python:** You need to have Python 3.10 installed on your system. You can download Python from the official website: [Python Downloads](https://www.python.org/downloads/).

2. **Dependencies:** This script relies on certain Python libraries and packages. You can install them by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

## MySQL Server: 
You must have access to a MySQL server where the target database resides. If you need to create the MySQL server manually, please follow your organization's guidelines or the database provider's instructions.

## Configuration
Before running the script, you need to make some specific configurations:

Environment Variables (`.env`)
In the root directory of the project, you should create a file named `.env`. This file should contain the following environment variables, specific to your database setup:

    
    DB_SERVER=your_server_name
    DB_NAME=your_database_name
    DB_USER=your_username
    DB_PASSWORD=your_password
    

## Directory for Uploads
This script may involve file uploads. You need to specify the directory where uploaded files will be stored. Ensure that this directory exists and is writable. You can configure the upload directory in the script's code, usually by modifying a variable or parameter.
    
    UPLOAD_FOLDER='<your-directory>'
    app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
    

## Running the Script
To run the script, open your command line or terminal and navigate to the directory where the script is located. Then, execute the following command:

    flask --app test.py --debug run

## Access the Website
You may access the website through the link [http://127.0.0.1:5000](http://127.0.0.1:5000) or the link displayed in your terminal.

## Troubleshooting 
If you are denied access to your own localhost, click the following link: chrome://net-internals/#sockets and click on 'flush socket pools', and re-run your flask application. 
