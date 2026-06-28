import { createSignal, createEffect } from 'solid-js';

export type Theme = 'light' | 'dark';

// dark is always default - only respects a previously saved value
const getInitialTheme = (): Theme => {
  const saved = localStorage.getItem('docgraph-theme') as Theme | null;
  if (saved === 'light' || saved === 'dark') return saved;
  return 'dark';
};

// reactive signal - exported so components can read the current theme
const [theme, setTheme] = createSignal<Theme>(getInitialTheme());

// applies 'light' class to <html> and persists choice - css :root defines dark by default
createEffect(() => {
  const current = theme();
  const root = document.documentElement;
  if (current === 'light') {
    root.classList.add('light');
    root.classList.remove('dark');
  } else {
    root.classList.remove('light');
    root.classList.add('dark');
  }
  localStorage.setItem('docgraph-theme', current);
});

// flips between dark and light
export const toggleTheme = () =>
  setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));

// sets a specific theme value directly
export const setThemeExplicit = (t: Theme) => setTheme(t);

export { theme };
