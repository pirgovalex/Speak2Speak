import { Router, Route } from '@solidjs/router';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import AudioPlayer from './components/AudioPlayer';

function Layout(props) {
  return (
    <div class="chat-layout">
      <Sidebar />
      {props.children}
      <AudioPlayer />
    </div>
  );
}

function Welcome() {
  return (
    <div class="flex-1 flex flex-col items-center justify-center p-8 text-center text-secondary">
      <div class="mb-4 text-4xl font-light text-slate-300">Speak2Speak</div>
      <p>Select a chat or start a new conversation.</p>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Route path="/" component={Layout}>
        <Route path="/" component={Welcome} />
        <Route path="/chat/:id" component={ChatWindow} />
      </Route>
    </Router>
  );
}

export default App;
