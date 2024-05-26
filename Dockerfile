# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Turns off multi-thread - solves "OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option."
ENV OMP_NUM_THREADS=1

# Solve "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose streamlit port
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
