#use python 3.9 as base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# copy the requirements file into the container
COPY requirements.txt /app/

#install dependancies
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the Flask app port
EXPOSE 5000

# Define the command to run the Flask app
CMD [ "python", "app.py" ]