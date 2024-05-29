FROM fschlatt/natural-language-processing-exercises:0.0.1

# Add the run.py script
ADD run.py /code/run.py

# Add the model_lib.py file which contains the model, vectorizer, and best threshold
ADD svc_model.joblib /code/svc_model.joblib
ADD vectorizer.joblib /code/vectorizer.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]
