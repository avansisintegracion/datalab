CREATE DATABASE demo_ind;
USE demo_ind;
CREATE TABLE random (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  first_letter CHAR(1) NOT NULL,
  first_letter_index CHAR(1) NOT NULL,
  string VARCHAR(10) NOT NULL,
  value SMALLINT UNSIGNED NOT NULL,
  value_index SMALLINT UNSIGNED NOT NULL,
  PRIMARY KEY (id),
  INDEX ind_first_letter (first_letter_index(1)),
  INDEX ind_value (value_index)
  )
  ENGINE=INNODB;