import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { EncryptionInterfaceComponent } from './encryption-interface/encryption-interface.component';

const routes: Routes = [
  { path: '', component: EncryptionInterfaceComponent },
  // Puedes agregar más rutas aquí si es necesario
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
