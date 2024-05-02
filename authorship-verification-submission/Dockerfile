# docker build -t username/authorship-verification-logistic-regression:0.0.1 .
FROM python:3.8-slim

WORKDIR /code

RUN pip install numpy pandas scikit-learn joblib tira

COPY . /code

RUN python train.py

ENTRYPOINT ["python3", "run.py"]