FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD svc_model.pkl /code/svc_model.pkl
ADD vectorizer.pkl /code/vectorizer.pkl
ADD best_threshold.pkl /code/best_threshold.pkl

ENTRYPOINT [ "python3", "/code/run.py" ]
