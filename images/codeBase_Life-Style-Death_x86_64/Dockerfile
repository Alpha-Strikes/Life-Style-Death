# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest

WORKDIR /app
#install runtime deps
RUN pip install --no-cache-dir pandas numpy scikit-learn statsmodels

#create directories for code and activation data
RUN mkdir -p /app/codeBase /acGvaGonBase

#copy inference code and README
COPY images/codeBase_Life-Style-Death_x86_64/run_inference.py /app/codeBase/
COPY images/codeBase_Life-Style-Death_x86_64/README.md /app/codeBase/

#copy activation data (required by docker-compose to copy to volume)
COPY data/activation_data.csv /acGvaGonBase/activation_data.csv

CMD ["python", "/app/codeBase/run_inference.py"]