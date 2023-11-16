# dsa3101-2310-12-ocr
Private Git Repository for DSA3101 AY23/24 Sem 1, Group 12-ocr

## General Information
The What-A-Waste Web project is a groundbreaking initiative dedicated to waste data management and analysis at NUS bin centers, offering valuable insights and actionable opportunities. This project harnesses Optical Character Recognition (OCR) technology to enhance data accuracy, enabling our website to collect and store data effectively and generate meaningful dashboards through a user-friendly interface.

## Prerequisite
1. **Docker**: Ensure Docker desktop is installed on your system. If not, follow the official [Docker installation guide](https://docs.docker.com/get-docker/).

2. **Python:** You need to have Python 3.10 installed on your system. You can download Python from the official website: [Python Downloads](https://www.python.org/downloads/).


## Repository structure
```plaintext
dsa3101-2310-12-ocr/
│
├── backend/ # Source code for the project
│   ├── dashboard/ # Forlder containing dashboard code
│   ├── database/ # Folder containing database & flask application
│   └── model/ # Folder containing OCR model code
│
├── frontend/ # Frontend early development (has been integrated into backend/ folder) 
│
├── docker-compose.yml # Docker compose file for the project
│
└──  README.md # README file for the project
```

## Setup Instructions
1. Navigate to your desired local directory, and proceed to clone the project repository into your local machine:
```bash
git clone https://github.com/wenjieng/dsa3101-2310-12-ocr.git
```

2. Enter into the root folder of the project ```dsa3101-2310-12-ocr```:
```bash
cd dsa3101-2310-12-ocr
```

3. Run the following command on the terminal to to start the project using Docker Compose:
```bash
docker compose up -d
```
This command initializes the project services defined in the ```docker-compose.yml``` and runs them in detached mode (```-d```), allowing the project to operate in the background.

4. Open your internet browser and visit http://localhost:9001/. 

## Additional notes
Login credentials: 
Access the web application using the provided credentials found in ```users.csv``` file. This file can be found in ```./backend/database``` directory.

ERR: Empty Response Error:
For Mac Users, during the uploading of images, you would face this error. Please use a Windows device if you would like to use this function.

## Troubleshooting
1. Problem: Denied access to your own localhost<br/>
Solution:<br/> 
1.1. Go to this website:  
```bash
chrome://net-internals/#sockets
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2. Click on 'flush socket pools'.<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.3. Re-run your flask application.


2. Problem: Python 3.10 Certificate issue<br/>
Solution: Run this command in the terminal 
```bash
/Applications/Python\ 3.10/Install\ Certificates.command
```
