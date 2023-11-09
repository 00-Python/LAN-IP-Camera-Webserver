#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Please provide a commit message as an argument."
  exit 1
fi

# Initialize migration directory if not already initialized
flask db init

# Create migration script with the provided commit message
flask db migrate -m "$1"

# Apply the migration script
flask db upgrade

# Exit with success status
exit 0
