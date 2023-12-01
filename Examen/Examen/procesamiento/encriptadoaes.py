import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def aes_encrypt(input_data, segment_size=1024):
    # Inicialización manual del contexto CUDA
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()

    try:
        # Definir el código del kernel de encriptación
        aes_kernel = """
        __global__ void aes_encrypt(char *input, char *output, char *key, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                char k = key[idx % 16]; // Clave fija de 16 bytes
                output[idx] = (input[idx] + k) ^ 0xAA; // Suma con la clave y XOR con un valor constante
            }
        }
        """

        # Compilar el kernel
        mod = SourceModule(aes_kernel)
        aes_encrypt_function = mod.get_function("aes_encrypt")

        # Definir una clave constante (16 bytes)
        clave_fija = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10'
        key_gpu = cuda.mem_alloc(16)
        cuda.memcpy_htod(key_gpu, clave_fija)

        # Procesar los datos en segmentos
        encrypted_result = bytearray()
        for i in range(0, len(input_data), segment_size):
            segment = input_data[i:i + segment_size]

            # Preparar los datos de entrada y salida para CUDA
            input_data_gpu = cuda.mem_alloc(len(segment))
            cuda.memcpy_htod(input_data_gpu, segment)

            output_data_gpu = cuda.mem_alloc(len(segment))
            output_data = np.zeros_like(segment)

            # Ejecutar el kernel de encriptación para el segmento
            aes_encrypt_function(input_data_gpu, output_data_gpu, key_gpu, np.int32(len(segment)),
                                 block=(256, 1, 1), grid=(int(len(segment)/256 + 1), 1))
            cuda.memcpy_dtoh(output_data, output_data_gpu)

            encrypted_result.extend(output_data)

            # Liberar memoria de la GPU para este segmento
            input_data_gpu.free()
            output_data_gpu.free()

        return encrypted_result

    finally:
        # Limpiar el contexto CUDA
        context.pop()
        context.detach()

# Función para aplicar la encriptación AES
def apply_aes_encrypt(input_data):
    return aes_encrypt(input_data.encode())

