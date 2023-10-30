setwd("C:/Users/wenji/Desktop/NUS Academics/NUS Y3S1/DSA3101/dsa3101-2310-12-ocr/backend/dashboard")

library(shiny)
library(ggplot2)
library(ggrepel)
library(DT)

library(viridis)
library(dplyr)
library(tidyverse)
library(lubridate)

df = read.csv("test_data.csv")

#Pre-processing
df_processed = df
df_processed$start_date = date(parse_date_time(df$start_date, "dmy", tz = "Singapore"))
df_processed$expiry_date = date(parse_date_time(df$expiry_date, "dmy", tz = "Singapore"))

df_processed = 
  df_processed %>%
  arrange(start_date, match(location, c("Start",
                                        "UTown Residences",
                                        "Tembusu / Cinnamon College",
                                        "College of Alice and Peter Tan",
                                        "Residential College 4",
                                        "Yale-NUS Cendana College",
                                        "Yale-NUS Elm College"))) %>%
  group_by(start_date) %>%
  mutate(calculated_weight_kg = gross_weight_kg-lag(gross_weight_kg, 1)) %>%
  ungroup() %>% 
  drop_na(calculated_weight_kg)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("What-a-Waste"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      #Date Range for Bar Chart
      dateRangeInput("dates_bar",
                     "Date range for Bar Chart",
                     start = "2023-10-01",
                     end = as.character(Sys.Date())),
      textOutput("DateRange_bar"),
      actionButton("goButton_bar", "Go!"),
      #Select group by criteria
      # selectInput("group", "Group by:",
      #             c("Day" = "cyl",
      #               "Month" = "am",
      #               "Year" = "gear",
      #               "Day of Week" = "",
      #               "Day of Month" = "",
      #               "Month of Year" = "")),
      #Filter date by Semester
      # selectInput("group", "Group by:",
      #             c("Semester 1" = "cyl",
      #               "Vacation 1" = "am",
      #               "Semester 2" = "gear",
      #               "vacation 2"= "")),
      #Filter date by Quarter of Year
      # selectInput("group", "Group by:",
      #             c("Q1" = "cyl",
      #               "Q2" = "am",
      #               "Q3" = "gear",
      #               "Q4" = "")),
      div(style="margin-bottom:10px"),
      #Date Range for Line Plot
      dateRangeInput("dates_line",
                     "Date range for Line Graph",
                     start = "2023-10-01",
                     end = as.character(Sys.Date())),
      textOutput("DateRange_line"),
      actionButton("goButton_line", "Go!"),
      div(style="margin-bottom:10px"),
      #Date Range for Tabular Data
      dateRangeInput("dates_table",
                     "Date range for Table",
                     start = "2023-10-01",
                     end = as.character(Sys.Date())),
      textOutput("DateRange_table"),
      actionButton("goButton_table", "Go!"),
      #Fix Position of SideBarPanel on Screen
      style = "position:fixed;width:30%;"
    ),
    
    # Show plots of 1x2 bar chart + line graph, and another row of tabular data
    mainPanel(fluidRow(
                verticalLayout(splitLayout(cellWidths = c("50%", "50%"),
                                           plotOutput("barPlot"),
                                           plotOutput("linePlot")),
                               DT::dataTableOutput("tablePlot"),
                               fluid = TRUE)
              )
    )
  )
)

# Define server logic required to draw a necessary plots/datatable
server <- function(input, output) {
  output$DateRange_table <- renderText({
    validate(
      need(input$dates_table[2] > input$dates_table[1], "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  output$tablePlot <- DT::renderDataTable({
    #execute when "Go!" is clicked
    input$goButton_table
    isolate({
      ##filters
      table_selected_start_date = date(parse_date_time(input$dates_table[1], "ymd", tz = "Singapore"))
      table_selected_end_date = date(parse_date_time(input$dates_table[2], "ymd", tz = "Singapore"))
      table_df_processed_filtered = 
        df_processed %>%
        filter(between(start_date, table_selected_start_date, table_selected_end_date))
      
      ##Summarise by location
      location_df = read.csv("location_database.csv")
      
      dashboard_tabular = 
        table_df_processed_filtered %>% 
        group_by(location, .drop = FALSE) %>% 
        summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
      
      dashboard_tabular_with_loc = 
        location_df %>%
        inner_join(dashboard_tabular, by = c("bin_centre" = "location"))
      
      ##table
      DT::datatable(dashboard_tabular_with_loc,
                    options = list(paging = TRUE,
                                   scrollX = FALSE,
                                   scrollY = TRUE,
                                   autoWidth = TRUE,
                                   buttons = c('csv', 'excel'),
                                   dom = 'Bfrtip'),
                    extensions = 'Buttons',
                    selection = 'multiple',
                    filter = 'bottom',
                    rownames = FALSE
      )
    })
  })
  output$DateRange_bar <- renderText({
    validate(
      need(input$dates_bar[2] > input$dates_bar[1], "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  output$barPlot <- renderPlot({
    #execute when "Go!" is clicked
    input$goButton_bar
    isolate({
      ##filters
      bar_selected_start_date = date(parse_date_time(input$dates_bar[1], "ymd", tz = "Singapore"))
      bar_selected_end_date = date(parse_date_time(input$dates_bar[2], "ymd", tz = "Singapore"))
      bar_df_processed_filtered = 
        df_processed %>%
        filter(between(start_date, bar_selected_start_date, bar_selected_end_date))
      
      ##plot
      ggplot(data = bar_df_processed_filtered,
             aes(x = start_date, y = calculated_weight_kg, fill = location)) +
        scale_fill_brewer(palette = "Spectral") +
        geom_bar(stat = 'identity', position = "dodge") + 
        labs(x = "Date",
             y = "Weight (kg)",
             title = "Total weight of General Waste collected by Bin Centre") +
        guides(fill = guide_legend(title = "Bin Centre")) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
        geom_text(aes(label = calculated_weight_kg), 
                  vjust = -0.4, 
                  position = position_dodge(0.9), 
                  size = 3.5,
                  fontface = "bold")
    })
  })
  output$DateRange_line <- renderText({
    validate(
      need(input$dates_line[2] > input$dates_line[1], "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  output$linePlot <- renderPlot({
    #execute when "Go!" is clicked
    input$goButton_line
    isolate({
      ##filters
      line_selected_start_date = date(parse_date_time(input$dates_line[1], "ymd", tz = "Singapore"))
      line_selected_end_date = date(parse_date_time(input$dates_line[2], "ymd", tz = "Singapore"))
      #line_selected_end_date = Sys.Date()
      line_df_processed_filtered = 
        df_processed %>%
        filter(between(start_date, line_selected_start_date, line_selected_end_date))
      
      ##Summarise by date
      dashboard_line = 
        line_df_processed_filtered %>% 
        group_by(start_date, .drop = FALSE) %>% 
        summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
      
      ##plot
      ggplot(data = dashboard_line, 
             aes(x = start_date, y = `Total Generated Waste (kg)`)) + 
        geom_line(linewidth = 1.5, color = 'darkgray') +  
        geom_point(size = 3, color = 'darkgray') +
        geom_text(aes(label = `Total Generated Waste (kg)`), 
                  #position = position_jitter(width = 0, height = 300), 
                  vjust = 2,
                  size = 3.5,
                  fontface = "bold") +
        ylim(c(min(dashboard_line$`Total Generated Waste (kg)`-200),max(dashboard_line$`Total Generated Waste (kg)`)))+
        labs(x = "Date",
             y = "Weight (kg)",
             title = "Total weight of General Waste collected by Date") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, face = "bold"))
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
