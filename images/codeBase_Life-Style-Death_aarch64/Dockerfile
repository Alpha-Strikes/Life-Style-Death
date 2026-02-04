# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest

WORKDIR /app
RUN pip install --no-cache-dir pandas numpy scikit-learn statsmodels

RUN mkdir -p /app/codeBase /acGvaGonBase

#reuse same inference script as x86_64
COPY images/codeBase_Life-Style-Death_x86_64/run_inference.py /app/codeBase/
COPY images/codeBase_Life-Style-Death_aarch64/README.md /app/codeBase/

COPY data/activation_data.csv /acGvaGonBase/activation_data.csv

CMD ["python3", "/app/codeBase/run_inference.py"]
