-- Drop the database if it exists
DROP DATABASE IF EXISTS OCR_DB;
CREATE DATABASE OCR_DB;
USE OCR_DB;

-- Drop the table if it exists
DROP TABLE IF EXISTS users, images CASCADE;

--
-- Table structure for table `users`
--

CREATE TABLE users (
  user_id INT(10) NOT NULL AUTO_INCREMENT,
  username VARCHAR(45) NOT NULL UNIQUE,
  password VARCHAR(45) NOT NULL,
  PRIMARY KEY (user_id, username)
  );

--
-- Table structure for table `images`
--

CREATE TABLE images (
  image_id INT(10) NOT NULL AUTO_INCREMENT,
  image LONGBLOB NOT NULL,
  start_date DATE NOT NULL,
  expiry_date DATE NOT NULL,
  location VARCHAR(45) NOT NULL,
  username VARCHAR(45) NOT NULL,
  gross_weight INT(10) NOT NULL,
  PRIMARY KEY (image_id)
  );

INSERT INTO users(username, password) VALUES('user1', 'password1');
INSERT INTO users(username, password) VALUES('user2', 'password2');
INSERT INTO users(username, password) VALUES('user3', 'password3');