#!/bin/bash

DB_DIR="db"
GEOMETRY3K_DIR="${DB_DIR}/Geometry3K"
GEOMETRY3K_LOGIC_FORMS_DIR="${DB_DIR}/Geometry3K_logic_forms"
TEMP_DIR="${DB_DIR}/temp_extracted"

mkdir -p "$GEOMETRY3K_DIR"
mkdir -p "$GEOMETRY3K_LOGIC_FORMS_DIR"
mkdir -p "$TEMP_DIR"

unzip "${DB_DIR}/train.zip" -d "$TEMP_DIR"
mv ${TEMP_DIR}/*/* "$GEOMETRY3K_DIR"

unzip "${DB_DIR}/val.zip" -d "$TEMP_DIR"
mv ${TEMP_DIR}/*/* "$GEOMETRY3K_DIR"

unzip "${DB_DIR}/test.zip" -d "$TEMP_DIR"
mv ${TEMP_DIR}/*/* "$GEOMETRY3K_DIR"

unzip "${DB_DIR}/logic_forms.zip" -d "$TEMP_DIR"
mv ${TEMP_DIR}/*/* "$GEOMETRY3K_LOGIC_FORMS_DIR"

rm -rf "$TEMP_DIR"

echo "All files have been successfully extracted."
