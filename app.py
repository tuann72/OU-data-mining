from shiny import render, reactive
from shiny.express import input, ui
from shinywidgets import render_widget
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

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
          # Summarize the Target column to avoid plotting per-row
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
        ui.card_header("Dataset")

        ui.input_radio_buttons(  
        "radio_group",  
        "Method of Choice",  
          {  
              "d": "Decision Trees",  
              "n": "Naive Bayes",  
              "l": "Logistic Regression",  
          },  
        )
        
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

          classifier(style="single", method=input.radio_group(),input=list_of_inputs)
        
  with ui.nav_panel("Mass Input"):
    ui.input_file("m", "Upload csv of multiple records", accept=".csv")
    ui.br()

    ui.input_radio_buttons(  
      "radio_group2",  
      "Method of Choice",  
        {  
            "d": "Decision Trees",  
            "n": "Naive Bayes",  
            "l": "Logistic Regression",  
        },  
      )
    ui.input_action_button("mass_predict_button", "Mass Prediction/Classification")  
    @reactive.effect
    @reactive.event(input.mass_predict_button)
    def massPredictBtn():
        fileInfo = input.m()
        file_path = fileInfo[0]["datapath"]
        df = pd.read_csv(file_path)
        list_of_inputs = df.values.tolist()
        classifier(style="multi", method=input.radio_group2(),input=list_of_inputs)
  
    ui.input_action_button("export_button", "Export Result as CSV")
    @reactive.effect
    @reactive.event(input.export_single_button)
    def exportSingleBtn():
        return f"{input.export_single_button()}"

#-------------------------- Visualizations Tab 
  with ui.nav_panel("Visualizations"):

    with ui.layout_columns():
      @render.plot
      def confusion_log():
        dat = np.loadtxt("log_cm.csv", delimiter=",", dtype=int)
        figure, axis = plt.subplots(figsize=(4,4))
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        axis.set_title("Logistic Regression", fontsize=12)
        plt.tight_layout()
        return figure
      
      @render.plot
      def confusion_dec():
        dat = np.loadtxt("log_cm.csv", delimiter=",", dtype=int)
        figure, axis = plt.subplots(figsize=(4,4))
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        axis.set_title("Decision Tree", fontsize=12)
        plt.tight_layout()
        return figure
      
      @render.plot
      def confusion_nav():
        dat = np.loadtxt("log_cm.csv", delimiter=",", dtype=int)
        figure, axis = plt.subplots(figsize=(4,4))
        display = ConfusionMatrixDisplay(confusion_matrix=dat,display_labels=["Dropout", "Non-Dropout"])
        display.plot(ax=axis, cmap="Greens", values_format="d", colorbar=False)
        axis.set_title("Naive Bayes", fontsize=12)
        plt.tight_layout()
        return figure


    with ui.layout_columns():
      @render.plot
      def plotPR():
        log_dat = pd.read_csv("logstic_pr.csv")
        dec_dat = pd.read_csv("decisiontree_pr.csv")
        nai_dat = pd.read_csv("naivebayes_pr.csv")
        plt.plot(log_dat["m_rec"].values, log_dat["m_pre"].values, label=f'Logistic Regression = {log_dat["auc_pr"].values[0]: .3f}')
        plt.plot(dec_dat["m_rec"].values, dec_dat["m_pre"].values, label=f'Decision Trees  = {dec_dat["auc_pr"].values[0]: .3f}')
        plt.plot(nai_dat["m_rec"].values, nai_dat["m_pre"].values, label=f'Naive Bayes = {nai_dat["auc_pr"].values[0]: .3f}')
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
  if style == "single":
    print(input)
  elif style == "multi":
    print(input)