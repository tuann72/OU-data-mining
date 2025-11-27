from shiny import render, reactive
from shiny.express import input, ui
from shinywidgets import render_widget
import plotly.express as px
import pandas as pd

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
        {"1": "Single", "2": "Married", "3": "Widowed"},  
        )
        ui.input_selectize(  
        "input2",  
        "Daytime/evening attendence",  
        {"1": "Daytime", "2": "Evening"},  
        )
        ui.input_selectize(  
        "input3",  
        "Previous qualification",  
        {"1": "Secondary Education", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input4",  
        "Nationality",  
        {"1": "Portuguese", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input5",  
        "Mother Qualification",  
        {"1": "Secondary Education", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input6",  
        "Father Qualification",  
        {"1": "Secondary Education", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input7",  
        "?",  
        {"1": "Choice A", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input8",  
        "?",  
        {"1": "Choice A", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input9",  
        "?",  
        {"1": "Choice A", "2": "Choice B", "3": "Choice C"},  
        )
        ui.input_selectize(  
        "input10",  
        "?",  
        {"1": "Choice A", "2": "Choice B", "3": "Choice C"},  
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
             input.input10()
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
    
  with ui.nav_panel("Visualizations"):
     pass

def classifier(style, method, input):
  if style == "single":
    pass
  elif style == "multi":
    pass