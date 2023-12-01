import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class EncryptionService {

  constructor(private http: HttpClient) { }

  rsaEncrypt(data: FormData) {
    return this.http.post('http://127.0.0.1:8000/rsa_encrypt/', data, { responseType: 'blob' });
  }

  rsaDecrypt(data: FormData) {
    return this.http.post('http://127.0.0.1:8000/rsa_decrypt/', data, { responseType: 'blob' });
  }

  aesEncrypt(data: FormData) {
    return this.http.post('http://127.0.0.1:8000/aes_encrypt/', data, { responseType: 'blob' });
  }

  aesDecrypt(data: FormData) {
    return this.http.post('http://127.0.0.1:8000/aes_decrypt/', data, { responseType: 'blob' });
  }
}
