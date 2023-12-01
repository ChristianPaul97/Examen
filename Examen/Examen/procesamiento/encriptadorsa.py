from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
import math

def text_to_ascii(text):
    return [ord(char) for char in text]

def encrypt_rsa(input_text, segment_size=1024):
    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)  # Suponiendo que estás usando la primera GPU
    context = device.make_context()

    try:
        # Convertir el texto de entrada a un array de NumPy y dividirlo en segmentos
        ascii_data = text_to_ascii(input_text)
        encrypted_result = []

        for i in range(0, len(ascii_data), segment_size):
            segment = ascii_data[i:i + segment_size]
            data_np = np.array(segment, dtype=np.int32)

            # Preparar memoria en la GPU para el segmento
            data_gpu = cuda.mem_alloc(data_np.nbytes)
            cuda.memcpy_htod(data_gpu, data_np)

            # Código del kernel de CUDA para la encriptación RSA
            kernel_code = """
            __device__ int modular_exponentiation(int base, int exponent, int modulus) {
                int result = 1;
                while (exponent > 0) {
                    if (exponent % 2 == 1)
                        result = (result * base) % modulus;
                    exponent = exponent >> 1;
                    base = (base * base) % modulus;
                }
                return result;
            }

            __global__ void rsa_encrypt_decrypt(int *data, int key, int n, int data_length) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < data_length) {
                    data[idx] = modular_exponentiation(data[idx], key, n);
                }
            }
            """

            # Compilar el kernel
            mod = SourceModule(kernel_code)
            rsa_kernel = mod.get_function("rsa_encrypt_decrypt")

            # Claves RSA y configuración de CUDA (valores fijos)
            e = 17  # Clave pública (e, n)
            n = 3233  # n = p * q, donde p y q son primos
            threads_per_block = 256
            blocks = math.ceil(len(segment) / threads_per_block)

            # Ejecutar el kernel para el segmento
            rsa_kernel(data_gpu, np.int32(e), np.int32(n), np.int32(len(segment)),
                       block=(threads_per_block, 1, 1), grid=(blocks, 1))

            # Recuperar datos encriptados del segmento
            encrypted_segment = np.empty_like(data_np)
            cuda.memcpy_dtoh(encrypted_segment, data_gpu)
            encrypted_result.extend(encrypted_segment.tolist())

            # Liberar memoria de la GPU para este segmento
            data_gpu.free()

        return encrypted_result

    finally:
        # Limpiar el contexto CUDA
        context.pop()

# Ejemplo de uso
input_text = "Hola, este es un mensaje de prueba."
encrypted_text = encrypt_rsa(input_text)
print(encrypted_text)
