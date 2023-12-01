# Usar la imagen base de NVIDIA CUDA
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Actualizar e instalar dependencias
RUN apt-get update -qq && apt-get install -qq -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    python3-pip &&\
    rm -rf /var/lib/apt/lists/*

# Instalar PyCUDA, Django, numpy, opencv-python, Pillow y django-cors-headers
# Añadir aquí otras dependencias específicas de tu proyecto si las hay
RUN pip3 install pycuda django numpy Pillow django-cors-headers==4.3.1

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar el proyecto al contenedor
COPY . /app

# Exponer el puerto en el que se ejecutará la aplicación Django
EXPOSE 8000

# Establecer variables de entorno
ENV PYTHONUNBUFFERED 1

# Comando para ejecutar el servidor Django
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
