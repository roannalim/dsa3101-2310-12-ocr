## General Information
We have enabled translation functions and implemented it in the web interface with the help of Flask-Babel. This `README.md` file includes the instructions of how to set up and update the translation files.

## Configuration
Before setting up the translations, you need to make some specific configurations:
1. Create a configuration file and save it as `babel.cfg` under path `dsa3101-2310-12-ocr/backend/database`. This file will contain these code:
```python
[python: app.py]
[jinja2: templates/**.html] 
extensions=jinja2.ext.i18n
```
2. (Optional) The code above assumes translation of all the HTML files. If you only want to translate and debug one of the HTML files, you can follow the following code instead:
```python
[python: app.py]
[jinja2: templates/<Specific_File_Name>.html]
extensions=jinja2.ext.i18n
```

## Setup Instructions
1. Install Flask-Babel:
```python
pip install flask-babel 
```

2. Open the HTML file that you want to enable the translation, formatting all the text that you want to be translated into this form. Ensure that the punctuations are whitespaces are accurate as shown below:
```HTML
{{ _('The text you want to be translated') }}
```

3. Extract Messages, please make sure your current working directory is `dsa3101-2310-12-ocr/backend/database`:
```bash
pybabel extract -F babel.cfg -o messages.pot .
```
Run this command in the terminal to extract messages from the source code. It creates a template file `messages.pot`.

4. Initialize Language Files:
For each language you want to support (replace `en` with your desired language codes like such as `zh` and `ms`):
```bash
pybabel init -i messages.pot -d translations -l en
```
This initializes language-specific `.po` files in the translations directory.

5. Editing `.po` Files:
Open each `.po` file in the translations directory and provide translations for the messages. These files contain message strings and their translations. One example shown below:
```python
#: templates/view_images.html:124
msgid "Logout"
msgstr "<The text you want to translate to, eg. 登出>"
```

6. Compile Translations, please make sure your current working directory is `dsa3101-2310-12-ocr/backend/database`:
```bash
pybabel compile -d translations
```
This command compiles the translated messages into binary `.mo` files, which Flask-Babel uses for runtime translations.

7. Update Translations:
This step is taken when you update your translations from the previous version. Repeat steps 2-6 to update the `messages.pot` file and the respective `.po` files.

