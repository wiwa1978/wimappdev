FROM python:3.9-slim

WORKDIR /code

RUN apt-get update && apt-get install -y

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY . /code

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]