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
  drop_na(calculated_weight_kg) %>%
  mutate(month_year = format(start_date, "%b %Y")) %>%
  mutate(year = as.integer(year(start_date))) %>%
  mutate(wday = wday(start_date, label = TRUE, abbr = TRUE)) %>%
  mutate(day = day(start_date)) %>%
  mutate(month = month(start_date, label = TRUE, abbr = TRUE)) %>%
  mutate(semester = case_when(
    (month == "Jan") | (month == "Feb") | (month == "Mar") | (month == "Apr") ~ "s1",
    (month == "May") | (month == "Jun") | (month == "Jul") ~ "v1",
    (month == "Aug") | (month == "Sep") | (month == "Oct") | (month == "Nov") ~ "s2",
    (month == "Dec") ~ "v2"
  )) %>%
  mutate(quarter = case_when(
    (month == "Jan") | (month == "Feb") | (month == "Mar") ~ "q4",
    (month == "Apr") | (month == "May") | (month == "Jun") ~ "q1",
    (month == "Jul") | (month == "Aug") | (month == "Sep") ~ "q2",
    (month == "Oct") | (month == "Nov") | (month == "Dec") ~ "q3"
  ))

# Define UI for application 
ui <- fluidPage(
  
  # Application title
  titlePanel("What-A-Waste"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel("Dashboard Configurations",
      #Date Range for Bar Chart
      dateRangeInput("dates_graph",
                     "Date range for Bar Chart & Line Plot",
                     start = "2023-10-01",
                     end = as.character(Sys.Date())),
      textOutput("DateRange_graph"),
      # Select "group by" criteria
      selectInput("group", "Group by:",
                  choices = c("Day" = "start_date",
                              "Month" = "month_year",
                              "Year" = "year",
                              "Day of Week" = "wday",
                              "Day of Month" = "day",
                              "Month of Year" = "month"),
                  selected = "start_date"),
      splitLayout(cellWidths = c("50%", "50%"),
        # Filter date by Semester
        checkboxGroupInput("semester", "Semester:",
                           choiceNames = 
                             list("Semester 1 (Jan - Apr)", "Vacation 1 (May - Jul)", "Semester 2 (Aug - Nov)", "Vacation 2 (Dec)"),
                           choiceValues = 
                             list("s1", "v1", "s2", "v2"),
                           selected = list("s1", "v1", "s2", "v2")),
        # Filter date by Quarter of Year
        checkboxGroupInput("quarter", "Quarter:",
                           choiceNames = 
                             list("Q1 (Apr - Jun)", "Q2 (Jul - Sep)", "Q3 (Oct - Dec)", "Q4 (Jan - Mar)"),
                           choiceValues = 
                             list("q1", "q2", "q3", "q4"),
                           selected = list("q1", "q2", "q3", "q4"))),
      actionButton("goButton_graph", "Apply Configurations for Graphs"),
      div(style="margin-bottom:30px"),
      #Date Range for Tabular Data
      dateRangeInput("dates_table",
                     "Date range for Table",
                     start = "2023-10-01",
                     end = as.character(Sys.Date())),
      textOutput("DateRange_table"),
      actionButton("goButton_table", "Apply Configurations for Table"),
      #Fix Position of SideBarPanel on Screen
      style = "position:fixed;width:30%;"
    ),
    
    # Show plots of 1x2 bar chart + line graph, and another row of tabular data
    mainPanel(fluidPage(
                verticalLayout(wellPanel(splitLayout(cellWidths = c("50%", "50%"),
                                           plotOutput("barPlot"),
                                           plotOutput("linePlot"))),
                               wellPanel(DT::dataTableOutput("tablePlot")),
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
    #execute when "Apply Configurations for Table" is clicked
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
        inner_join(dashboard_tabular, by = c("bin_centre" = "location")) %>%
        rename(id = location_id, Campus = campus, Precinct = precinct, "Bin Centre" = bin_centre)
      
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
  output$DateRange_graph <- renderText({
    validate(
      need(input$dates_graph[2] > input$dates_graph[1], "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  output$barPlot <- renderPlot({
    #execute when "Apply Configurations for Graphs" is clicked
    input$goButton_graph
    isolate({
      ##filters
      bar_selected_start_date = date(parse_date_time(input$dates_graph[1], "ymd", tz = "Singapore"))
      bar_selected_end_date = date(parse_date_time(input$dates_graph[2], "ymd", tz = "Singapore"))
      bar_df_processed_filtered = 
        df_processed %>%
        filter(between(start_date, bar_selected_start_date, bar_selected_end_date)) %>%
        filter(semester %in% input$semester) %>%
        filter(quarter %in% input$quarter)
      
      ##plot_function
      plot_bar <- function(input_group, xlab) { #for categorical x-axis
        ggplot(data = bar_df_processed_filtered,
               aes(x = input_group, y = calculated_weight_kg, fill = location)) +
          scale_fill_brewer(palette = "Spectral") +
          geom_bar(stat = 'identity', position = "dodge") + 
          labs(x = xlab,
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
      }
      
      plot_bar_v2 <- function(input_group, xlab) { #for continuous x-axis
        ggplot(data = bar_df_processed_filtered,
               aes(x = input_group, y = calculated_weight_kg, fill = location)) +
          scale_fill_brewer(palette = "Spectral") +
          geom_bar(stat = 'identity', position = "dodge") + 
          labs(x = xlab,
               y = "Weight (kg)",
               title = "Total weight of General Waste collected by Bin Centre") +
          guides(fill = guide_legend(title = "Bin Centre")) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
          geom_text(aes(label = calculated_weight_kg), 
                    vjust = -0.4, 
                    position = position_dodge(0.9), 
                    size = 3.5,
                    fontface = "bold") +
          scale_x_continuous(breaks = min(input_group):max(input_group))
      }
      
      #cater for user configuration + plot bar chart
      if (input$group == "month_year") {
        xlab = "Month & Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          mutate(month_year = factor(month_year, levels = unique(month_year))) %>%
          group_by(location, month_year, .drop = TRUE) %>%
          summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep")
        plot_bar(bar_df_processed_filtered$month_year, xlab)
      } else if (input$group == "year") {
        xlab = "Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, year, .drop = TRUE) %>%
          summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep")
        plot_bar_v2(bar_df_processed_filtered$year, xlab)
      } else if (input$group == "wday") {
        xlab = "Day of Week"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, wday, .drop = TRUE) %>%
          summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep")
        plot_bar(bar_df_processed_filtered$wday, xlab)
      } else if (input$group == "day") {
        xlab = "Day of Month"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, day, .drop = TRUE) %>%
          summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep")
        plot_bar_v2(bar_df_processed_filtered$day, xlab)
      } else if (input$group == "month") {
        xlab = "Month of Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, month, .drop = TRUE) %>%
          summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep")
        plot_bar(bar_df_processed_filtered$month, xlab)
      } else { #group_by(start_date) selected, smallest granularity, no need to group
        xlab = "Date"
        plot_bar(bar_df_processed_filtered$start_date, xlab)
        } 
    })
  })
  output$linePlot <- renderPlot({
    #execute when "Apply Configuration" is clicked
    input$goButton_graph
    isolate({
      ##filters
      line_selected_start_date = date(parse_date_time(input$dates_graph[1], "ymd", tz = "Singapore"))
      line_selected_end_date = date(parse_date_time(input$dates_graph[2], "ymd", tz = "Singapore"))
      line_df_processed_filtered =
        df_processed %>%
        filter(between(start_date, line_selected_start_date, line_selected_end_date)) %>%
        filter(semester %in% input$semester) %>%
        filter(quarter %in% input$quarter)
    
      ##plot_functions
      plot_line <- function(input_group, xlab) {
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste (kg)`)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste (kg)`),
                    #position = position_jitter(width = 0, height = 300),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(min(dashboard_line$`Total Generated Waste (kg)`-200),max(dashboard_line$`Total Generated Waste (kg)`)))+
          labs(x = xlab,
               y = "Weight (kg)",
               title = paste0("Total weight of General Waste collected by ", xlab)) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      }
      
      plot_line_v2 <- function(input_group, xlab) { #for continuous x-axis
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste (kg)`)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste (kg)`),
                    #position = position_jitter(width = 0, height = 300),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(min(dashboard_line$`Total Generated Waste (kg)`-200),max(dashboard_line$`Total Generated Waste (kg)`)))+
          labs(x = xlab,
               y = "Weight (kg)",
               title = paste0("Total weight of General Waste collected by ", xlab)) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
          scale_x_continuous(breaks = min(input_group):max(input_group))
      }
      
      plot_line_v3 <- function(input_group, xlab) { #for wday and month of year bc categorical variables
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste (kg)`, group = 1)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste (kg)`),
                    #position = position_jitter(width = 0, height = 300),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(min(dashboard_line$`Total Generated Waste (kg)`-200),max(dashboard_line$`Total Generated Waste (kg)`)))+
          labs(x = xlab,
               y = "Weight (kg)",
               title = paste0("Total weight of General Waste collected by ", xlab)) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      }
      
      #cater for user configuration + plot line graph
      if (input$group == "month_year") {
        xlab = "Month & Year"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(month_year = factor(month_year, levels = unique(month_year))) %>%
          group_by(month_year, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line_v3(dashboard_line$month_year, xlab)
      } else if (input$group == "year") {
        xlab = "Year"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(year, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line_v2(dashboard_line$year, xlab)
      } else if (input$group == "wday") {
        xlab = "Day of Week"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(wday = factor(wday, levels = unique(wday))) %>%
          group_by(wday, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line_v3(dashboard_line$wday, xlab)
      } else if (input$group == "day") {
        xlab = "Day of Month"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(day, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line_v2(dashboard_line$day, xlab)
      } else if (input$group == "month") {
        xlab = "Month of Year"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(month = factor(month, levels = unique(month))) %>%
          group_by(month, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line_v3(dashboard_line$month, xlab)
      } else { #group_by(start_date) selected, smallest granularity, no need to group
        xlab = "Date"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(start_date, .drop = FALSE) %>%
          summarise("Total Generated Waste (kg)" = sum(calculated_weight_kg), .groups = 'keep')
        plot_line(dashboard_line$start_date, xlab)
      } 
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
