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

        ui.input_checkbox_group(  
        "checkbox_group",  
        "Method of Choice",  
          {  
              "a": "Decision Trees",  
              "b": "Naive Bayes",  
              "c": "Logistic Regression",  
          },  
        )
        def value():
          return ", ".join(input.checkbox_group())
        
        ui.input_action_button("predict_button", "Predict")
        @reactive.event(input.predict_button)
        def predictBtn():
            return f"{input.predict_button()}"
        
  with ui.nav_panel("Mass Input"):
    ui.input_file("m", "Upload multiple records")
    @render.text
    def massTxt():
      return input.m()

    ui.input_action_button("mass_predict_button", "Mass Prediction/Classification")  
    @reactive.event(input.mass_predict_button)
    def massPredictBtn():
        return f"{input.mass_predict_button()}"
  
    ui.input_action_button("export_button", "Export Result as CSV")
    @reactive.event(input.export_single_button)
    def exportSingleBtn():
        return f"{input.export_single_button()}"
    
  with ui.nav_panel("Visualizations"):
     pass




