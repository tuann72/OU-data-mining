# OU-data-mining

We utilize Conda for our Python environments

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Then we can run the application.

```bash
shiny run --reload app.py
```

We can view the site at http://127.0.0.1:8000/