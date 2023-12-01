# Examen
# Aceleración de Cifrado y Descifrado de Datos con PyCUDA

## Descripción
Este proyecto implementa un sistema acelerado de cifrado y descifrado de datos utilizando PyCUDA para aprovechar las capacidades de cálculo de las GPUs. 
Proporciona una interfaz web construida con Django que permite a los usuarios cifrar y descifrar archivos de texto. 
El sistema soporta los algoritmos criptográficos AES y RSA y está contenerizado con Docker para facilitar su despliegue y escalabilidad.

## Prerrequisitos
- Python 3.x
- Django
- PyCUDA
- Docker
- GPU con soporte CUDA

## Instalación
Clona el repositorio en tu máquina local:
git clone [https://github.com/tu-github/tu-proyecto.git](https://github.com/ChristianPaul97/Examen.git)

css
Copy code

Navega al directorio del proyecto:
cd tu-proyecto

yaml
Copy code

Construye los contenedores Docker:
docker-compose build

shell
Copy code

## Uso
Ejecuta los contenedores Docker:
docker-compose up

## Interfaz Web
La interfaz web permite a los usuarios:
- Subir archivos de texto para cifrado y descifrado.
- Elegir entre los métodos de cifrado AES y RSA.
- Descargar archivos cifrados o descifrados.

## Dockerización
La configuración de Docker incluye contenedores separados para la aplicación Django, PyCUDA y el frontend Angular. Utiliza los archivos `Dockerfile` y `docker-compose.yml` proporcionados para la gestión de contenedores.

## Licencia
Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

## Colaboradores
- Christian Portoviejo
- Adrian Lopez

Siéntete libre de contribuir a este proyecto mediante la presentación de pull requests.

## Agradecimientos
Nos gustaría agradecer a todos los que apoyaron y contribuyeron al desarrollo de este proyecto.

## Contacto
Para cualquier consulta, por favor contáctanos en:
- cportoviejo@est.ups.edu.ec
- alopeza9@est.ups.edu.ec
