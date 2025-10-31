# smile-extraction

a package to extract and process data collected from monkey host in the Batista lab

## Documentation

Docs are built with Sphinx and auto-deployed to GitHub Pages.

- Live site: https://raeedcho.github.io/smile-extraction/
- Sources in `docs/` with `conf.py`.

### Build locally

Install docs extras and build HTML:

```bash
pip install .[doc]
sphinx-build -b html docs _build/html
```

Open `_build/html/index.html` in your browser.