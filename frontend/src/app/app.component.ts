import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ChatComponent } from './pages/chat/chat.component';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ChatComponent],
  template: `
    <router-outlet></router-outlet>
  `
})
export class AppComponent {
  title = 'stage';
}
