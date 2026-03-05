# AI Fundamentals - Project Instructions

## Git Commit Rules
- NEVER add "Co-Authored-By" lines to commit messages
- NEVER credit Claude as a contributor or co-author in commits
- Write concise, descriptive commit messages focused on what changed and why
- Follow conventional commit style matching existing repo history

## Project Structure
- Single-page HTML files with embedded CSS/JS (zero dependencies)
- Deployed on GitHub Pages from `main` branch
- Root `index.html` is the landing page with chapter grid
- Each chapter lives in its own directory (e.g., `function-explorer/`)

## Design System (matches portfolio at /Users/mahesh/portfolio)
- Fonts: Outfit (display), Source Sans 3 (body), JetBrains Mono (mono)
- Light mode: #FDF6F3 bg, #FC6A49 coral accent, #232F3E text
- Dark mode: #1A2332 bg, same coral accent, #E2E8F0 text
- Cards: 16-20px border-radius, subtle shadows, 3D hover transforms
- MLU-style buttons with offset shadow layer

## Tech Stack
- Pure vanilla HTML, CSS, JavaScript
- No frameworks, no build tools, no npm
- SVG for visualizations
- Google Fonts via CDN
