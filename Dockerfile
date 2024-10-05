FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["jupyter", "nbconvert", "--to", "python", "Team2_FinalProject_EXL.ipynb"]
