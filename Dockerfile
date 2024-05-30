FROM python:3.9-slim

RUN pip install sympy matplotlib

COPY main.py /app/main.py
WORKDIR /app

ENTRYPOINT ["python", "main.py"]
