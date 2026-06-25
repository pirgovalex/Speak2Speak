/** @type {import('tailwindcss').Config} */
module.exports = {
  // Use 'class' strategy — .light class on <html> enables light theme
  // (Dark is the default via :root CSS custom properties)
  darkMode: 'class',

  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './index.html',
  ],

  theme: {
    extend: {
      colors: {
        // Medical-grade muted color palette
        brand: {
          primary: '#5A7A9A',    // Muted steel blue — accent
          hover: '#4A6A8A',      // Slightly darker hover state
          subtle: '#1A2535',     // Dark subtle background tint
        },
        surface: {
          'dark-900': '#111111', // Near-black background (dark default)
          'dark-800': '#1A1A1A', // Slightly lighter panels
          'dark-700': '#1C1C1C', // Card backgrounds
          'dark-600': '#2A2A2A', // Borders
          'light-50':  '#F5F5F5', // Near-white background
          'light-100': '#FFFFFF', // Pure white panels/cards
          'light-200': '#E0E0E0', // Light borders
        },
        text: {
          'dark-primary':   '#E8E8E8',
          'dark-secondary': '#A0A0A0',
          'dark-muted':     '#666666',
          'light-primary':   '#1A1A1A',
          'light-secondary': '#555555',
          'light-muted':     '#999999',
        },
        chat: {
          'user-bg-dark':   '#1E2D3D',
          'user-text-dark': '#D0E4F0',
          'ai-bg-dark':     '#1C1C1C',
          'ai-text-dark':   '#DCDCDC',
          'user-bg-light':  '#E8F0F7',
          'user-text-light': '#1A2A3A',
          'ai-bg-light':    '#F0F0F0',
          'ai-text-light':  '#1A1A1A',
        },
        status: {
          'success-dark': '#4A8C6A',
          'warning-dark': '#8C7A3A',
          'error-dark':   '#8C3A3A',
          'success-light': '#3A7A5A',
          'warning-light': '#7A6A2A',
          'error-light':   '#7A2A2A',
        },
      },

      borderRadius: {
        // Sharp-edge design language — maximum 2px
        DEFAULT: '0',
        sm: '2px',
        md: '2px',
        lg: '2px',
        xl: '2px',
        '2xl': '2px',
        full: '2px',
      },

      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },

      boxShadow: {
        card: '0 1px 3px rgba(0, 0, 0, 0.5)',
        elevated: '0 4px 12px rgba(0, 0, 0, 0.6)',
        'card-light': '0 1px 3px rgba(0, 0, 0, 0.08)',
        'elevated-light': '0 4px 12px rgba(0, 0, 0, 0.12)',
      },

      backgroundImage: {
        // No gradients — flat design only
        none: 'none',
      },
    },
  },

  plugins: [],
};
