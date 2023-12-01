import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EncryptionInterfaceComponent } from './encryption-interface.component';

describe('EncryptionInterfaceComponent', () => {
  let component: EncryptionInterfaceComponent;
  let fixture: ComponentFixture<EncryptionInterfaceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [EncryptionInterfaceComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EncryptionInterfaceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
