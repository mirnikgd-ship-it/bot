# Используем Python 3.10 (самый стабильный для NLP задач)
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем сам скрипт парсера
COPY main.py .

# Запускаем скрипт
CMD ["python", "-u", "main.py"]