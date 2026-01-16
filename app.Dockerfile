FROM python:3.11.8-slim-bullseye

EXPOSE 8501

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

EXPOSE 8501

# Run main.py when the container launches using Streamlit
CMD ["streamlit", "run", "app.py"]
