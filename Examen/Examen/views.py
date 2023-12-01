import base64
from django.http import HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .procesamiento import encriptadorsa, desencriptadorsa, encriptadoaes, desencriptadoaes
import json  


def create_file_response(file_content, file_name):
    response = HttpResponse(file_content, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename="{}"'.format(file_name)
    return response

@csrf_exempt
@require_http_methods(["POST"])
def encrypt_viewrsa(request):
    try:
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file provided")
        file = request.FILES['file']
        input_text = file.read().decode('utf-8')
        
        encrypted_text = encriptadorsa.encrypt_rsa(input_text)

        # Convertir la lista de números a una cadena JSON
        encrypted_text_json = json.dumps(encrypted_text)

        return create_file_response(encrypted_text_json, "encrypted_rsa.txt")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

@csrf_exempt
@require_http_methods(["POST"])
def decrypt_viewrsa(request):
    try:
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file provided")
        file = request.FILES['file']
        encrypted_data = json.loads(file.read().decode('utf-8'))

        if not isinstance(encrypted_data, list) or not all(isinstance(item, int) for item in encrypted_data):
            return HttpResponseBadRequest("Invalid encrypted data format")

        decrypted_text = desencriptadorsa.decrypt_rsa(encrypted_data)
        return create_file_response(decrypted_text, "decrypted_rsa.txt")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

@csrf_exempt
@require_http_methods(["POST"])
def aes_encrypt_view(request):
    try:
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file provided")
        file = request.FILES['file']
        input_text = file.read().decode('utf-8')

        encrypted_data = encriptadoaes.apply_aes_encrypt(input_text)
        encrypted_data_base64 = base64.b64encode(encrypted_data).decode('utf-8')

        return create_file_response(encrypted_data_base64, "encrypted_aes.txt")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

@csrf_exempt
@require_http_methods(["POST"])
def aes_decrypt_view(request):
    try:
        if 'file' not in request.FILES:
            return HttpResponseBadRequest("No file provided")
        file = request.FILES['file']
        encrypted_text_base64 = file.read().decode('utf-8')

        encrypted_data = base64.b64decode(encrypted_text_base64)
        decrypted_text = desencriptadoaes.decrypt_aes(encrypted_data)

        return create_file_response(decrypted_text, "decrypted_aes.txt")
    except Exception as e:
        return HttpResponseBadRequest(str(e))
