FROM python:3.7-slim

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py app.py
COPY model.h5 model.h5

EXPOSE 8000

CMD ["python", "app.py"]