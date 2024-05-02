# docker build -t username/authorship-verification-logistic-regression:0.0.1 .
FROM python:3.8-slim

RUN pip install numpy pandas scikit-learn joblib

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

ENTRYPOINT ["python3", "/code/run.py"]