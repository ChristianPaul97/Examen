from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
import math

def ascii_to_text(ascii_list):
    return ''.join(chr(num) for num in ascii_list)

def decrypt_rsa(encrypted_data, segment_size=4096):
    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)  # Suponiendo que estás usando la primera GPU
    context = device.make_context()

    try:
        decrypted_text = ''
        for i in range(0, len(encrypted_data), segment_size):
            # Procesar cada segmento
            segment = encrypted_data[i:i + segment_size]
            encrypted_data_np = np.array(segment, dtype=np.int32)

            # Preparar memoria en la GPU para el segmento
            data_gpu = cuda.mem_alloc(encrypted_data_np.nbytes)
            cuda.memcpy_htod(data_gpu, encrypted_data_np)

            # Código del kernel de CUDA para la desencriptación RSA
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

            # Claves RSA y configuración de CUDA
            d = 2753  # Clave privada (d, n)
            n = 3233  # n = p * q, donde p y q son primos
            threads_per_block = 256
            blocks = math.ceil(len(segment) / threads_per_block)

            # Ejecutar el kernel para el segmento
            rsa_kernel(data_gpu, np.int32(d), np.int32(n), np.int32(len(segment)),
                       block=(threads_per_block, 1, 1), grid=(blocks, 1))

            # Recuperar y concatenar los resultados
            decrypted_segment = np.empty_like(encrypted_data_np)
            cuda.memcpy_dtoh(decrypted_segment, data_gpu)
            decrypted_text += ascii_to_text(decrypted_segment)

            # Liberar memoria de la GPU para este segmento
            data_gpu.free()

        return decrypted_text

    finally:
        # Limpiar el contexto CUDA
        context.pop()
