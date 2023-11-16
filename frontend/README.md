## General Information
This folder is mainly for the front-end members to test the flask application, web interface design, and connections among web pages at the early stage of the project. After the back-end team set up the database and OCR model, we integrated our designs into the templates located in the `../backend/database` folder.

## Repository Structure

```plaintext
dsa3101-2310-12-ocr/
│
├── frontend/ # Frontend early development (has been integrated into ../backend/database folder) 
│   ├── static/ # Folder containing images used by web pages
│   ├── templates/ # Folder containing web pages to be served by Flask application, written in HTML & CSS
│   ├── translations/ # Folder containing translation strings used by web pages
│   ├── Web Images/ # Folder containing backup for images used by web pages
│   ├── app.py # Flask application to serve the web pages
│   ├── babel.cfg # Configuration file for Flask-Babel
│   ├── message.pot # Contain translatable strings for web pages
│   └── user.csv # User credential
```

## Running the Flask Application
1. Ensuring the current directory is in `./frontend`, run our application using flask:
	```bash
	flask --app app.py --debug run
	```
2. Open your internet browser and visit [http://127.0.0.1:5000](http://127.0.0.1:5000) or the link given in the terminal.

## Multi-language Support
We have enabled translation functions and implemented it in the web interface with the help of Flask-Babel. At the early stage, we only managed to set up the webpage in Mandarin and English.

### Configuration
Before setting up the translation, you need to make some specific configurations:
1. Create a configuration file and save it as `babel.cfg`. The file will contain these code:
```python
[python: app.py]
[jinja2: templates/**.html] 
extensions=jinja2.ext.i18n
```
2. The code above assumes translation of all the HTML files. If you only want to translate and debug one of the HTML files, you can follow the following code instead:
```python
[python: app.py]
[jinja2: templates/<Specific_File_Name>.html]
extensions=jinja2.ext.i18n
```

### Setup Instructions
1. Install Flask-Babel:
```python
pip install flask-babel 
```

2. Open the HTML file that you want to enable the translation, formatting all the text that you want to be translated into this form. Ensure that the punctuations are whitespaces are accurate as shown below:
```HTML
{{ _('The text you want to be translated') }}
```

3. Extract Messages:
```bash
pybabel extract -F babel.cfg -o messages.pot .
```
Run this command in the terminal to extract messages from the source code. It creates a template file `messages.pot`.

4. Initialize Language Files:
For each language you want to support (replace `en` with your desired language codes like such as `zh` and `ms`):
```bash
pybabel init -i messages.pot -d translations -l en
```
This initializes language-specific ```.po``` files in the translations directory.

5. Editing `.po` Files:
Opening each `.po` file in the translations directory and provide translations for the messages. These files contain message strings and their translations. One example shown below:
```python
#: templates/view_images.html:124
msgid "Logout"
msgstr "<The text you want to translate to, eg. 你好>"
```

6. Compile Translations:
```bash
pybabel compile -d translations
```
This command compiles the translated messages into binary `.mo` files, which Flask-Babel uses for runtime translations.

7. Update Translations:
This step is taken when you update your translations from the previous version. Repeat steps 2-6 to update the `messages.pot` file and the respective `.po` files.
