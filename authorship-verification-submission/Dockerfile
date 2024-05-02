# docker build -t username/authorship-verification-logistic-regression:0.0.1 .
FROM python:3.8-slim

RUN pip install numpy pandas scikit-learn joblib tira

ADD train.py /code/train.py
ADD run.py /code/run.py

RUN python /code/train.py
ENTRYPOINT ["python3", "/code/run.py"]
