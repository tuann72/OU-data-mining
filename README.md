# OU-data-mining

We utilize Conda for our Python environments

```bash
conda create -n myenv python=3.10
conda activate myenv
conda install --file requirements.txt
```

Then we can run the application.

```bash
shiny run --reload app.py
```

We can view the site at http://127.0.0.1:8000/