import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def decrypt_aes(encrypted_data, segment_size=1024):
    # Inicializar CUDA
    cuda.init()
    device = cuda.Device(0)  # Suponiendo que estás usando la primera GPU
    context = device.make_context()

    try:
        aes_kernel = """
        __global__ void aes_decrypt(char *input, char *output, char *key, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                char k = key[idx % 16]; // Clave fija de 16 bytes
                output[idx] = (input[idx] ^ 0xAA) - k; // XOR con un valor constante y resta de la clave
            }
        }
        """

        mod = SourceModule(aes_kernel)
        aes_decrypt = mod.get_function("aes_decrypt")

        # Definir una clave constante (16 bytes)
        clave_fija = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10'
        key_gpu = cuda.mem_alloc(16)
        cuda.memcpy_htod(key_gpu, clave_fija)

        decrypted_result = bytearray()
        for i in range(0, len(encrypted_data), segment_size):
            segment = encrypted_data[i:i + segment_size]
            input_data = np.frombuffer(segment, dtype=np.uint8)
            decrypted_data = np.zeros_like(input_data)

            input_data_gpu = cuda.mem_alloc(input_data.nbytes)
            cuda.memcpy_htod(input_data_gpu, input_data)

            decrypted_data_gpu = cuda.mem_alloc(input_data.nbytes)

            aes_decrypt(input_data_gpu, decrypted_data_gpu, key_gpu, np.int32(len(input_data)),
                        block=(256, 1, 1), grid=(int(len(input_data)/256 + 1), 1))
            cuda.memcpy_dtoh(decrypted_data, decrypted_data_gpu)

            decrypted_result.extend(decrypted_data)

            # Liberar memoria de la GPU para este segmento
            input_data_gpu.free()
            decrypted_data_gpu.free()

        return decrypted_result.decode('utf-8', errors='ignore')

    finally:
        # Limpiar el contexto CUDA
        context.pop()
