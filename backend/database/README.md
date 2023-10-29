# README

## General Information

This Python script is designed to perform a specific task that involves interacting with a SQL server and managing a database. Here, we will provide an overview of the script, its purpose, and the steps required to run it successfully.

### Database Information

This script connects to a SQL server to manipulate data within a specific database. The details of the database, such as the server name, database name, and credentials, need to be configured in the `.env` file, which will be explained in the next section.

## Prerequisites

Before running this Python script, make sure you have the following prerequisites in place:

1. **Python:** You need to have Python 3.10 installed on your system. You can download Python from the official website: [Python Downloads](https://www.python.org/downloads/).

2. **Dependencies:** This script relies on certain Python libraries and packages. You can install them by running the following command:

   
   pip install -r requirements.txt
   

## MySQL Server: 
You must have access to a MySQL server where the target database resides. If you need to create the MySQL server manually, please follow your organization's guidelines or the database provider's instructions.

## Configuration
Before running the script, you need to make some specific configurations:

Environment Variables (.env)
In the root directory of the project, you should create a file named .env. This file should contain the following environment variables, specific to your database setup:

    
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
    
## Troubleshooting 
If you are denied access to your own localhost, click the following link: chrome://net-internals/#sockets and click on 'flush socket pools', and re-run your flask application. 