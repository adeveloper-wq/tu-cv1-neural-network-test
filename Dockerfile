FROM tensorflow/tensorflow:latest
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD [ "python","-u","./script.py" ] 