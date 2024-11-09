/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'cs2': {
          'primary': '#FF4655',
          'secondary': '#0F1923',
        }
      }
    },
  },
  plugins: [],
}