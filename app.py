from shiny import render, reactive
from shiny.express import input, ui
from shinywidgets import render_widget
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import decision_trees.helper as dt
import naive_bayes.helper as nb
from datetime import datetime

### TEST DATA FOR FIGURES
df = pd.read_csv("clean_data.csv")

#-------------------------- Title
ui.page_opts(title="Analysis of Student Factors For Dropout Rates", fillable=True)    
    

#-------------------------- Main Body
file_name = "clean_data.csv"
test = pd.read_csv(file_name)
with ui.navset_card_tab(id="tab"):
  with ui.nav_panel("Data Preview"):
    with ui.layout_columns():
      # Target Display
      @render_widget
      def plot():
          # Summarize the Target column
          summary = df["Target"].value_counts().reset_index()
          summary.columns = ["Target", "Count"]

          label_correction = {
             0 : "Dropout",
             1 : "Non-dropout"
          }
          
          summary["Target"] = summary["Target"].map(label_correction)

          # Create pie chart
          fig = px.pie(
              data_frame=summary,
              names="Target",
              values="Count",
              title="Student Classifications"
          ).update_layout(title_x=0.5)

          return fig
      
      # Meta Data
      with ui.card():  
        ui.card_header("Meta Data")
        ui.p("File Name: " + file_name)
        ui.p("Number of records: " + str(len(df)))
        ui.p("Number of features: " + str(len(df.columns) - 1))

    with ui.card():  
        ui.card_header("Dataset")
        @render.data_frame
        def test_df():
            return render.DataGrid(df)

#-------------------------- Prediction Tab
  with ui.nav_panel("Prediction/Classification"):
    with ui.layout_columns():
      with ui.card():  
        ui.card_header("Input Fields")
        # example input ui component
        ui.input_selectize(
          "input1",
          "Marital Status",
          {"1": "Single", "2": "Married", "3": "Widower", "4": "Divorced", "5": "Facto union", "6": "Legally separated"}
        )

        ui.input_selectize(
          "input2",
          "Daytime/Evening Attendance",
          {"1": "Daytime", "0": "Evening"}
        )

        ui.input_selectize(
          "input3",
          "Nationality",
          {"1": "Portuguese", "2": "German", "6": "Spanish", "11": "Italian", "13": "Dutch", "14": "English", "17": "Lithuanian", "21": "Angolan", "22": "Cape Verdean", "24": "Guinean", "25": "Mozambican", "26": "Santomean", "32": "Turkish", "41": "Brazilian", "62": "Romanian", "100": "Moldova (Republic of)", "101": "Mexican", "103": "Ukrainian", "105": "Russian", "108": "Cuban", "109": "Colombian"}
        )

        ui.input_selectize(
          "input4",
          "Mother's Qualification",
          {"1": "Secondary Education-12th Year", "2": "Higher Education-Bachelor's", "3": "Higher Education -Degree", "4": "Higher Education -Master's", "5": "Higher Education -Doctorate", "6": "Frequency of Higher Education", "9": "12th Year - Not Completed", "10": "11th Year - Not Completed", "11": "7th Year (Old)", "12": "Other - 11th Year", "14": "10th Year", "18": "General commerce course", "19": "Basic Education 3rd Cycle", "22": "Technical-professional course", "26": "7th year", "27": "2nd cycle general high school", "29": "9th Year - Not Completed", "30": "8th year", "34": "Unknown", "35": "Can't read or write", "36": "Can read without 4th year", "37": "Basic education 1st cycle", "38": "Basic Education 2nd Cycle", "39": "Technological specialization", "40": "Higher education - degree (1st cycle)", "41": "Specialized higher studies", "42": "Professional higher technical", "43": "Higher Education - Master (2nd cycle)", "44": "Higher Education - Doctorate (3rd cycle)"}
        )

        ui.input_selectize(
          "input5",
          "Father's Qualification",
          {"1": "Secondary Education-12th Year", "2": "Higher Education-Bachelor's", "3": "Higher Education-Degree", "4": "Higher Education-Master's", "5": "Higher Education - Doctorate", "6": "Frequency of Higher Education", "9": "12th Year - Not Completed", "10": "11th Year - Not Completed", "11": "7th Year (Old)", "12": "Other - 11th Year", "13": "2nd year complementary high school", "14": "10th Year", "18": "General commerce course", "19": "Basic Education 3rd Cycle", "20": "Complementary High School", "22": "Technical-professional course", "25": "Complementary High School - not concluded", "26": "7th year", "27": "2nd cycle general high school", "29": "9th Year - Not Completed", "30": "8th year", "31": "General Course Administration and Commerce", "33": "Supplementary Accounting and Administration", "34": "Unknown", "35": "Can't read or write", "36": "Can read without 4th year", "37": "Basic education 1st cycle", "38": "Basic Education 2nd Cycle", "39": "Technological specialization", "40": "Higher education - degree (1st cycle)", "41": "Specialized higher studies", "42": "Professional higher technical", "43": "Higher Education - Master (2nd cycle)", "44": "Higher Education - Doctorate (3rd cycle)"}
        )

        ui.input_selectize(
          "input6",
          "Mother's Occupation",
          {"0": "Student", "1": "Legislative/Executive Representatives", "2": "Specialists in Intellectual Activities", "3": "Intermediate Level Technicians", "4": "Administrative staff", "5": "Personal Services Workers", "6": "Agriculture/Fisheries Workers", "7": "Skilled Workers in Industry", "8": "Machine Operators", "9": "Unskilled Workers", "10": "Armed Forces", "90": "Other Situation", "99": "(blank)", "122": "Health professionals", "123": "Teachers", "125": "ICT Specialists", "131": "Science/engineering technicians", "132": "Health technicians", "134": "Legal/social/cultural technicians", "141": "Office workers", "143": "Data/accounting operators", "144": "Other administrative support", "151": "Personal service workers", "152": "Sellers", "153": "Personal care workers", "171": "Construction workers", "173": "Printing/precision workers", "175": "Food/woodworking workers", "191": "Cleaning workers", "192": "Unskilled agriculture workers", "193": "Unskilled industry workers", "194": "Meal preparation assistants"}
        )

        ui.input_selectize(
          "input7",
          "Father's Occupation",
          {"0": "Student", "1": "Legislative/Executive Representatives", "2": "Specialists in Intellectual Activities", "3": "Intermediate Level Technicians", "4": "Administrative staff", "5": "Personal Services Workers", "6": "Agriculture/Fisheries Workers", "7": "Skilled Workers in Industry", "8": "Machine Operators", "9": "Unskilled Workers", "10": "Armed Forces", "90": "Other Situation", "99": "(blank)", "101": "Armed Forces Officers", "102": "Armed Forces Sergeants", "103": "Other Armed Forces", "112": "Directors administrative/commercial", "114": "Hotel/catering directors", "121": "Physical sciences specialists", "122": "Health professionals", "123": "Teachers", "124": "Finance/accounting specialists", "131": "Science/engineering technicians", "132": "Health technicians", "134": "Legal/social/cultural technicians", "135": "ICT technicians", "141": "Office workers", "143": "Data/accounting operators", "144": "Other administrative support", "151": "Personal service workers", "152": "Sellers", "153": "Personal care workers", "154": "Protection/security personnel", "161": "Farmers/agricultural workers", "163": "Subsistence farmers", "171": "Construction workers", "172": "Metallurgy workers", "174": "Electricity/electronics workers", "175": "Food/woodworking workers", "181": "Plant/machine operators", "182": "Assembly workers", "183": "Vehicle drivers", "192": "Unskilled agriculture workers", "193": "Unskilled industry workers", "194": "Meal preparation assistants", "195": "Street vendors"}
        )

        ui.input_selectize(
          "input8",
          "Displaced",
          {"1": "Yes", "0": "No"}
        )

        ui.input_selectize(
          "input9",
          "Educational Special Needs",
          {"1": "Yes", "0": "No"}
        )

        ui.input_selectize(
          "input10",
          "Debtor",
          {"1": "Yes", "0": "No"}
        )

        ui.input_selectize(
          "input11",
          "Tuition Fees Up to Date",
          {"1": "Yes", "0": "No"}
        )

        ui.input_selectize(
          "input12",
          "Gender",
          {"1": "Male", "0": "Female"}
        )

        ui.input_selectize(
          "input13",
          "Scholarship Holder",
          {"1": "Yes", "0": "No"}
        )

        ui.input_numeric(
          "input14",
          "Age at Enrollment",
          value=20,
          min=17,
          max=70
        )

        ui.input_selectize(
          "input15",
          "International",
          {"1": "Yes", "0": "No"}
        )

        ui.input_numeric(
          "input16",
          "Curricular Units 1st Sem (Credited)",
          value=0,
          min=0,
          max=20
        )

        ui.input_numeric(
          "input17",
          "Curricular Units 2nd Sem (Credited)",
          value=0,
          min=0,
          max=20
        )

        ui.input_numeric(
          "input18",
          "Unemployment Rate (%)",
          value=10.0,
          min=0,
          max=100,
          step=0.1
        )

        ui.input_numeric(
          "input19",
          "Inflation Rate (%)",
          value=2.0,
          min=-10,
          max=50,
          step=0.1
        )

        ui.input_numeric(
          "input20",
          "GDP",
          value=1.0,
          min=-10,
          max=10,
          step=0.1
        )

      with ui.card():  
        results  = reactive.value("")
        ui.card_header("Dataset")

        ui.input_radio_buttons(  
        "radio_group",  
        "Method of Choice",  
          {  
              "d": "Decision Trees",  
              "n": "Naive Bayes" 
          },  
        )
        
        # button with action listern for prediction
        ui.input_action_button("predict_button", "Predict")
        @reactive.effect
        @reactive.event(input.predict_button)
        def predictBtn():
          list_of_inputs = []

          # store all inputs into a list
          list_of_inputs.extend([
             input.input1(),
             input.input2(),
             input.input3(),
             input.input4(),
             input.input5(),
             input.input6(),
             input.input7(),
             input.input8(),
             input.input9(),
             input.input10(),
             input.input11(),
             input.input12(),
             input.input13(),
             input.input14(),
             input.input15(),
             input.input16(),
             input.input17(),
             input.input18(),
             input.input19(),
             input.input20(),
          ])

          # convert all values to ints
          list_of_inputs = list(map(float, list_of_inputs))

          # reactively set variable to render text
          results.set(classifier(style="single", method=input.radio_group(),input=list_of_inputs))
        
        # render results
        @render.text
        def single_result():
          return f"Prediction: {results()}"
        
  with ui.nav_panel("Mass Input"):
    # store confrimation reactively
    confirmation = reactive.value("")
    # store dataframe to display reactively
    frame = reactive.value(pd.DataFrame())
    # upload ui component accepting csv
    ui.input_file("m", "Upload csv of multiple records", accept=".csv")
    ui.br()

    # ui options to choose a method
    ui.input_radio_buttons(  
      "radio_group2",  
      "Method of Choice",  
        {  
            "d": "Decision Trees",  
            "n": "Naive Bayes"  
        },  
      )

    # button with listerner to mass predict
    ui.input_action_button("mass_predict_button", "Mass Prediction/Classification")  
    @reactive.effect
    @reactive.event(input.mass_predict_button)
    def massPredictBtn():
        # load in file data
        fileInfo = input.m()
        # store file path
        file_path = fileInfo[0]["datapath"]
        # open up file path and store as dataframe
        df = pd.read_csv(file_path)
        # convert records into a list of list
        list_of_inputs = df.values.tolist()
        # mass classification
        mass_res = classifier(style="multi", method=input.radio_group2(),input=list_of_inputs)
        df["Classification"] = mass_res
        frame.set(df)
    
    # render data frame as a table when prediction is completed
    @render.data_frame
    def createFrameResult():
      return frame()
  
    # export button
    ui.input_action_button("export_button", "Export Result as CSV")
    @reactive.effect
    @reactive.event(input.export_button)
    def exportSingleBtn():
      dat = frame()
      # export dataframe as csv
      dat.to_csv("results.csv", index=False)
      confirmation.set("Saved as results.csv | Time: " + str(datetime.now()))

    # send confirmation text to inform user that it saved
    @render.text
    def mass_confirm():
      return confirmation()

#-------------------------- Visualizations Tab 
  with ui.nav_panel("Visualizations"):

    ### render confusion matrix
    with ui.layout_columns():
      @render.plot
      def confusion_log():
        # read confusion matrix results
        dat = np.loadtxt("log_cm.csv", delimiter=",", dtype=int)
        # create a 2x2 plot
        figure, axis = plt.subplots(figsize=(4,4))
        # create a confusion matrix display
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        # display plot
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        # title plot
        axis.set_title("Logistic Regression", fontsize=12)
        plt.tight_layout()
        return figure
      
      @render.plot
      def confusion_dec():
        dat = np.loadtxt("decisiontree_cm.csv", delimiter=",", dtype=int)
        figure, axis = plt.subplots(figsize=(4,4))
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        axis.set_title("Decision Tree", fontsize=12)
        plt.tight_layout()
        return figure
      
      @render.plot
      def confusion_nav():
        dat = np.loadtxt("naivebayes_cm.csv", delimiter=",", dtype=int)
        figure, axis = plt.subplots(figsize=(4,4))
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        axis.set_title("Naive Bayes", fontsize=12)
        plt.tight_layout()
        return figure

    # Create plots for Precision recall curve and ROC curve
    with ui.layout_columns():
      @render.plot
      def plotPR():
        # read in data points from each respective method
        log_dat = pd.read_csv("logstic_pr.csv")
        dec_dat = pd.read_csv("decisiontree_pr.csv")
        nai_dat = pd.read_csv("naivebayes_pr.csv")
        # plot the curves
        plt.plot(log_dat["m_rec"].values, log_dat["m_pre"].values, label=f'Logistic Regression = {log_dat["auc_pr"].values[0]: .3f}')
        plt.plot(dec_dat["m_rec"].values, dec_dat["m_pre"].values, label=f'Decision Trees  = {dec_dat["auc_pr"].values[0]: .3f}')
        plt.plot(nai_dat["m_rec"].values, nai_dat["m_pre"].values, label=f'Naive Bayes = {nai_dat["auc_pr"].values[0]: .3f}')
        # label plot
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend()
        plt.grid(True)

      @render.plot
      def plotROC():
        log_dat = pd.read_csv("logstic_roc.csv")
        dec_dat = pd.read_csv("decisiontree_roc.csv")
        nai_dat = pd.read_csv("naivebayes_roc.csv")
        plt.plot(log_dat["fp"].values, log_dat["tp"].values, label=f'Logistic Regression = {log_dat["auc_roc"].values[0]: .3f}')
        plt.plot(dec_dat["fp"].values, dec_dat["tp"].values, label=f'Decision Trees  = {dec_dat["auc_roc"].values[0]: .3f}')
        plt.plot(nai_dat["fp"].values, nai_dat["tp"].values, label=f'Naive Bayes = {nai_dat["auc_roc"].values[0]: .3f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

def classifier(style, method, input):
  results = ""
  # decide if method is for single prediction or multiple
  if style == "single":
    # decide if method is decision trees or navie bayes
    if method == "d":
      results = decisionTreePredict(input)
    else:
      results = naviePredict(input)
  elif style == "multi":
    if method == "d":
      results = decisionTreePredict(input, multiple=True)
    else:
      results = naviePredict(input, multiple=True)
  return results

# The methods were copied over from the folders decision_trees and naive_bayes,
# Further comments will be avalible in those files

def decisionTreePredict(input, multiple=False):
  data = np.loadtxt("clean_data.csv", delimiter=',', skiprows=1)
  # Get number of examples m and dimensionality n
  m, n = np.shape(data)

  # Split dataset into training and testing
  train_set, test_set = dt.split_data(data)

  categorical_indices = {i for i in range(n)}
  numerical_indices = {13, 15, 16, 17, 18, 19}
  categorical_indices = categorical_indices - numerical_indices

  tree = dt.construct_tree(train_set, [i for i in range(n-1)], categorical_indices)
  if(multiple == False):
    res = dt.predict(tree, input)
    if(res == 0):
      res = "Dropout"
    else:
      res = "Non-Dropout"
  else:
    res = []
    for record in input:
      temp = dt.predict(tree, record)
      if(temp == 0):
        temp = "Dropout"
      else:
        temp = "Non-Dropout"
      res.append(temp)

  return res

def naviePredict(input, multiple = False):
  data = np.loadtxt("clean_data.csv", delimiter=',', skiprows=1)
  m, n = np.shape(data)
  n -= 1
  num_neg, num_pos = 0, 0
  mu_neg, mu_pos = [0 for i in range(n)], [0 for i in range(n)]
  train_set, test_set = nb.split_data(data)

  # Mu values for continuous features
  for instance in train_set:
      if instance[n] > 0: 
          num_pos += 1
          for i in range(n):
              mu_pos[i] += instance[i]
      else:
          num_neg += 1
          for i in range(n):
              mu_neg[i] += instance[i]

  for i in range(n):
    mu_neg[i] = mu_neg[i] / num_neg
    mu_pos[i] = mu_pos[i] / num_pos

  # Sigma squared values for continuous features
  sigma_neg, sigma_pos = [0 for i in range(n)], [0 for i in range(n)]

  for instance in train_set:
      if instance[n] > 0: 
          for i in range(n):
              sigma_pos[i] += (instance[i] - mu_pos[i])**2
      else:
          for i in range(n):
              sigma_neg[i] += (instance[i] - mu_neg[i])**2

  # Conditional probs for categorical features
  for i in range(n):
      sigma_neg[i] = sigma_neg[i] / (num_neg-1)
      sigma_pos[i] = sigma_pos[i] / (num_pos-1)

  categorical_indices = {i for i in range(n)}
  numerical_indices = {13, 15, 16, 17, 18, 19}
  categorical_indices = categorical_indices - numerical_indices

  cat_counts_neg = [{} for i in range(n)]
  cat_counts_pos = [{} for i in range(n)]

  possible_values = {
      0: [1, 2, 3, 4, 5, 6], # Marital Status
      1: [0, 1], # Daytime/Evening Attendance
      2: [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109], # Nationality
      3: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], # Mothers Qualification
      4: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], # Fathers Qualification
      5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194], # Mothers Occupation
      6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195], # Fathers Occupation
      7: [0, 1], # Displaced
      8: [0, 1], # Educational Special Needs
      9: [0, 1], # Debtor
      10: [0, 1], # Tuition Fees Up to Date
      11: [0, 1], # Gender
      12: [0, 1], # Scholarship Holder
      14: [0, 1], # International
      20: [0, 1] # Target
  }

  for i in categorical_indices:
      for val in possible_values[i]:
          cat_counts_pos[i][val] = 0
          cat_counts_neg[i][val] = 0

  for instance in train_set:
      cls = 'pos' if instance[n] > 0 else 'neg'
      for i in categorical_indices:
          val = instance[i]
          # count occurrences for categorical
          d = cat_counts_pos[i] if cls == 'pos' else cat_counts_neg[i]
          d[val] = d.get(val, 0) + 1

  for i in categorical_indices:
      K = len(possible_values[i])
      for val in possible_values[i]:
          cat_counts_pos[i][val] = (cat_counts_pos[i][val] + 1) / (num_pos + K)
          cat_counts_neg[i][val] = (cat_counts_neg[i][val] + 1) / (num_neg + K)

  if(multiple == False):
    neg_prior, pos_prior = (num_neg / len(train_set)), (num_pos / len(train_set))
    prob_neg, prob_pos, prediction = nb.predict_naive(categorical_indices, n, pos_prior, neg_prior, input, mu_neg, mu_pos, sigma_neg, sigma_pos, cat_counts_neg, cat_counts_pos)
    if(prediction == 0):
      prediction = "Dropout"
    else:
      prediction = "Non-Dropout"
    return prediction
  else:
    predictions = []
    for record in input:
      # Reset negative and positive class probabilities for each instance
      neg_prior, pos_prior = (num_neg / len(train_set)), (num_pos / len(train_set))
      prob_neg, prob_pos, res = nb.predict_naive(categorical_indices, n, pos_prior, neg_prior, record, mu_neg, mu_pos, sigma_neg, sigma_pos, cat_counts_neg, cat_counts_pos)
      if(res == 0):
        pred = "Dropout"
      else:
         pred = "Non-Dropout"
      predictions.append(pred)
    return predictions