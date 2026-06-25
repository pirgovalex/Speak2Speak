import { createSignal, createEffect } from 'solid-js';

/**
 * Theme type — only two valid values
 */
export type Theme = 'light' | 'dark';

/**
 * getInitialTheme
 * Dark is always the default. Only respects a previously saved localStorage value.
 * Never falls back to the system preference — DocGraph is a dark-first application.
 */
const getInitialTheme = (): Theme => {
  const saved = localStorage.getItem('docgraph-theme') as Theme | null;
  if (saved === 'light' || saved === 'dark') return saved;
  return 'dark'; // Default is always dark
};

/**
 * theme — reactive signal for current theme
 * Exported so components can read it reactively.
 */
const [theme, setTheme] = createSignal<Theme>(getInitialTheme());

/**
 * Effect: apply/remove the 'light' class on <html> and persist to localStorage.
 * We use class="light" on :root (not "dark") because the CSS :root block defines
 * the dark theme by default, and .light overrides it.
 */
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

/**
 * toggleTheme — flip between dark and light
 */
export const toggleTheme = () =>
  setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));

/**
 * setThemeExplicit — set a specific theme value
 */
export const setThemeExplicit = (t: Theme) => setTheme(t);

export { theme };
