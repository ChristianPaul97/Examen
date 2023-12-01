import { Component } from '@angular/core';
import { EncryptionService } from '../encryption.service';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-encryption-interface',
  templateUrl: './encryption-interface.component.html',
  styleUrls: ['./encryption-interface.component.scss']
})
export class EncryptionInterfaceComponent {
  selectedFile: File | null = null;
  fileContent: string | null = null;  // Añade esta línea
  encryptedDecryptedContent: string | null = null; // Añade esta propiedad para almacenar el resultado
  downloadUrl: SafeUrl | null = null;

  constructor(private encryptionService: EncryptionService, private sanitizer: DomSanitizer) { }

  onFileSelected(event: Event) {
    const fileInput = event.target as HTMLInputElement;
    if (fileInput.files && fileInput.files.length > 0) {
      this.selectedFile = fileInput.files[0];
      this.readFile(this.selectedFile); // Llama a la función para leer el archivo
    }
  }

  encrypt(method: string) {
    this.downloadUrl = null;
    this.encryptedDecryptedContent = null; // Asegúrate de limpiar el contenido anterior
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);
      if (method === 'RSA') {
        this.encryptionService.rsaEncrypt(formData).subscribe(response => {
          this.handleResponse(response); // Cambio aquí
        });
      } else if (method === 'AES') {
        this.encryptionService.aesEncrypt(formData).subscribe(response => {
          this.handleResponse(response); // Cambio aquí
        });
      }
    }
  }
  
  decrypt(method: string) {
    this.downloadUrl = null;
    this.encryptedDecryptedContent = null; // Asegúrate de limpiar el contenido anterior
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);
      if (method === 'RSA') {
        this.encryptionService.rsaDecrypt(formData).subscribe(response => {
          this.handleResponse(response); // Cambio aquí
        });
      } else if (method === 'AES') {
        this.encryptionService.aesDecrypt(formData).subscribe(response => {
          this.handleResponse(response); // Cambio aquí
        });
      }
    }
  }
  
  // Nueva función para manejar la respuesta y actualizar la UI
  private handleResponse(data: Blob) {
    const reader = new FileReader();
    reader.onload = () => {
      this.encryptedDecryptedContent = reader.result as string; // Actualizamos la propiedad con el contenido
    };
    reader.readAsText(data); // Convertimos el Blob a texto
    this.handleDownload(data); // Continuamos con el manejo para la descarga
  }

  private handleDownload(data: Blob) {
    const blob = new Blob([data], { type: 'text/plain' });
    this.downloadUrl = this.sanitizer.bypassSecurityTrustUrl(window.URL.createObjectURL(blob));
  }
  

  private readFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      this.fileContent = reader.result as string;
    };
    reader.readAsText(file);
  }

  clearTextAreas() {
    // Resetea el contenido de las áreas de texto
    this.fileContent = null;
    this.encryptedDecryptedContent = null;
  }

}
