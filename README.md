# OU-data-mining

## Files

### File Details
dt_helper.py contains helper functions for decision trees that include splitting data, finding entropy, gain, optimal threshold, etc.

decisiontrees.py constructs the tree using the helper functions and plots the results

lr_helper.py contains helper functions for logistic regression such as data splitting, training the model, making predictions, etc.

logisticregression.py utilizes the lr_helper.py to create a logistic regression model and plots the results

nb_helper.py contains helper functions for naive bayes including splitting the data, calculating the conditional probabities, etc.

naviebayes.py performs the naive bayes classification using nb_helper functions to make predictions and plot results

app.py contains a Shiny web app that compiles a UI with interactions that rely on calling functions from dt_helper.py, lr_helper.py, nb_helper.py to perform classifications.

### Expected Structure to run
decision_trees (folder)
- decisiontrees.py
- dt_helper.py
logistic_regression (folder)
- logisticregression.py
- lr_helper.py
naive_bayes (folder)
- naivebayes.py
- nb_helper.py
app.py
clean_data.csv

## Running the application
Create Python environment and install requirements
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