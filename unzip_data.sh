#!/bin/bash

DB_DIR="db"
GEOMETRY3K_DIR="${DB_DIR}/Geometry3K"
GEOMETRY3K_LOGIC_FORMS_DIR="${DB_DIR}/Geometry3K_logic_forms"

mkdir -p "$GEOMETRY3K_DIR"
mkdir -p "$GEOMETRY3K_LOGIC_FORMS_DIR"

unzip "${DB_DIR}/train.zip" -d "$GEOMETRY3K_DIR"
unzip "${DB_DIR}/val.zip" -d "$GEOMETRY3K_DIR"
unzip "${DB_DIR}/test.zip" -d "$GEOMETRY3K_DIR"

unzip "${DB_DIR}/logic_forms.zip" -d "$GEOMETRY3K_LOGIC_FORMS_DIR"

echo "All files have been successfully extracted."
