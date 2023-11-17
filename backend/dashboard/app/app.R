library(shiny)
library(shinyWidgets)
library(ggplot2)
library(DT)

library(dplyr)
library(tidyverse)
library(lubridate)

library(RMySQL)

# Choose database flag = "test-data" or "database"
flag = "database"

if (flag == "test-data") {
  df = read.csv("test_data.csv")
} else {
  # MySQL Database Connection & Query
  ##delay startup by 30 seconds for MySQL database to start up completely
  Sys.sleep(30)
  establish_sql_connection <- function() {
    mysqlconnection = dbConnect(RMySQL::MySQL(),
                                host = "db",
                                user = "root",
                                password = "icebear123",
                                dbname = "OCR_DB")
    
    result = dbSendQuery(mysqlconnection,
                         "SELECT image_id, start_date, expiry_date, location, username, gross_weight FROM images")
    
    print(paste("Last retrieved from database at:", with_tz(Sys.time(), tzone = "Asia/Singapore"), "(SGT)"))
    
    df = dbFetch(result)
    
    dbDisconnect(mysqlconnection)
    print(df)
    return(df)
  }
  df = establish_sql_connection()
}

# Pre-processing of data and calculating weight at each bin centre
df_processing <- function(df_processed) {
  df_processed$start_date = date(parse_date_time(df_processed$start_date, "ymd", tz = "Singapore")) 
  df_processed$expiry_date = date(parse_date_time(df_processed$expiry_date, "ymd", tz = "Singapore"))
  
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
    mutate(calculated_weight_kg = gross_weight-dplyr::lag(gross_weight, 1)) %>%
    ungroup() %>% 
    drop_na(calculated_weight_kg) %>%
    mutate(calculated_weight_tonnes = calculated_weight_kg/1000) %>%
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
    )) %>%
    mutate(fiscal_year = ifelse(
      month(start_date)>3,
      paste0("FY-APR-", substring(year(start_date),3, 4)," - MAR-", substring(year(start_date)+1,3, 4)),
      paste0("FY-APR-", substring(year(start_date)-1,3, 4)," - MAR-", substring(year(start_date),3, 4))
    ))
  return(df_processed)
  }

df_processed = df_processing(df)

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
                                start = "2023-01-01",
                                end = as.character(Sys.Date())),
                 textOutput("DateRange_graph"),
                 # Select "group by" criteria
                 selectInput("group", "Group by:",
                             choices = c("Day" = "start_date",
                                         "Month" = "month_year",
                                         "Year" = "year",
                                         "Day of Week" = "wday",
                                         "Day of Month" = "day",
                                         "Month of Year" = "month",
                                         "Fiscal Year" = "fiscal_year"),
                             selected = "start_date"),
                 splitLayout(cellWidths = c("42%", "28%", "30%"), cellArgs = list(style = "height: 140px"), 
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
                                                selected = list("q1", "q2", "q3", "q4")),
                             # Allow selection of units of measurement for weight (kg or tonnes)
                             selectInput("weight_unit", "Display Weight in:", width = "80%",
                                         choices = c("Kg" = "calculated_weight_kg",
                                                     "Tonnes" = "calculated_weight_tonnes"))),
                 # Allow selection of bin centres for comparison
                 dropdownButton(label = "Select Bin Centre:", status = "default", width = 80, tooltip = TRUE, circle = FALSE,
                                checkboxGroupInput(inputId = "bin_centre_selection",label = "Choose",
                                                   choices = unique(df_processed$location),
                                                   selected = unique(df_processed$location))),
                 div(style="margin-bottom:10px"),
                 actionButton("goButton_graph", "Apply Configurations for Graphs"),
                 div(style="margin-bottom:20px"),
                 # Date Range for Tabular Data
                 dateRangeInput("dates_table",
                                "Date range for Table",
                                start = "2023-01-01",
                                end = as.character(Sys.Date())),
                 textOutput("DateRange_table"),
                 actionButton("goButton_table", "Apply Configurations for Table"),
                 div(style="margin-bottom:20px"),
                 actionButton("goButton_home", "Return to Home",
                              icon("house-user"),
                              onclick = "location.href='http://127.0.0.1:9001/home?';", 
                              style = 'display: inline-block; margin-right: 15px; float:right'),
                 actionButton("goButton_DataRefresh", "Refresh Data",
                              icon("arrows-rotate"),
                              style = 'display: inline-block; margin-right: 15px; float:right'),
                 # Fix Position of SideBarPanel on Screen
                 style = "position:fixed; width:33%;"
    ),
    
    # Show plots of 1x2 bar chart + line graph, and another row of tabular data
    mainPanel(fluidPage(
      verticalLayout(wellPanel(splitLayout(cellWidths = c("50%", "50%"),
                                           plotOutput("barPlot"),
                                           plotOutput("linePlot"))),
                     wellPanel(DT::dataTableOutput("tablePlot")),
                     fluid = TRUE)
    ))
  )
)

# Define server logic required to draw a necessary plots/datatable
server <- function(input, output, session) {
  if (flag == "database") { 
    counter = reactiveValues(countervalue = 0)
    observeEvent(input$goButton_DataRefresh, {counter$countervalue <- counter$countervalue + 1})
    df_refreshed <- eventReactive(
      #Refresh data pull from database when button is clicked
      input$goButton_DataRefresh, {
      req(establish_sql_connection())
      # Pre-processing of data and calculating weight at each bin centre
      df = establish_sql_connection()
      df_processing(df)
      }
    )
  }
  # print text when invalid date range
  output$DateRange_table <- renderText({
    end_table = date(parse_date_time(input$dates_table[2], "ymd", tz = "Singapore"))
    start_table = date(parse_date_time(input$dates_table[1], "ymd", tz = "Singapore"))
    validate(
      need(as.numeric(difftime(end_table, start_table, units = c("days"))) >= 0,
           "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  
  output$tablePlot <- DT::renderDataTable({
    #Refresh data from database
    if (flag == "database"){
      if (counter$countervalue != 0) {
        df_processed = df_refreshed()
      }
    }
    #execute when "Apply Configurations for Table" is clicked
    input$goButton_table
    isolate({
      ##filters
      table_selected_start_date = date(parse_date_time(input$dates_table[1], "ymd", tz = "Singapore"))
      table_selected_end_date = date(parse_date_time(input$dates_table[2], "ymd", tz = "Singapore"))
      table_df_processed_filtered = 
        df_processed %>%
        dplyr::filter(between(start_date, table_selected_start_date, table_selected_end_date)) %>%
        mutate(calculated_weight = case_when(
          input$weight_unit == "calculated_weight_kg" ~ calculated_weight_kg,
          input$weight_unit == "calculated_weight_tonnes" ~ calculated_weight_tonnes))
      
      ##summarise by location
      location_df = read.csv("location_database.csv")
      
      if (input$weight_unit == "calculated_weight_kg") {
        dashboard_tabular = 
          table_df_processed_filtered %>% 
          group_by(location, .drop = FALSE) %>% 
          summarise("Total Generated Waste (kg)" = sum(calculated_weight), .groups = 'keep')
      } else if (input$weight_unit == "calculated_weight_tonnes") {
        dashboard_tabular = 
          table_df_processed_filtered %>% 
          group_by(location, .drop = FALSE) %>% 
          summarise("Total Generated Waste (tonnes)" = sum(calculated_weight), .groups = 'keep')
      }
      
      dashboard_tabular_with_loc = 
        location_df %>%
        inner_join(dashboard_tabular, by = c("bin_centre" = "location")) %>%
        rename(id = location_id, Campus = campus, Precinct = precinct, "Bin Centre" = bin_centre)
      
      ##table visualisation
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
  
  # print text when invalid date range
  output$DateRange_graph <- renderText({
    end_graph = date(parse_date_time(input$dates_graph[2], "ymd", tz = "Singapore"))
    start_graph = date(parse_date_time(input$dates_graph[1], "ymd", tz = "Singapore"))
    validate(
      need(as.numeric(difftime(end_graph, start_graph, units = c("days"))) >= 0,
           "ERROR: End date is before Start date. Please re-select date range.")
    )
  })
  
  output$barPlot <- renderPlot({
    #Refresh data from database
    if (flag == "database"){
      if (counter$countervalue != 0) {
        df_processed = df_refreshed()
      }
    }
    #execute when "Apply Configurations for Graphs" is clicked
    input$goButton_graph
    isolate({
      ##filters
      bar_selected_start_date = date(parse_date_time(input$dates_graph[1], "ymd", tz = "Singapore"))
      bar_selected_end_date = date(parse_date_time(input$dates_graph[2], "ymd", tz = "Singapore"))
      bar_df_processed_filtered = 
        df_processed %>%
        dplyr::filter(between(start_date, bar_selected_start_date, bar_selected_end_date)) %>%
        dplyr::filter(semester %in% input$semester) %>%
        dplyr::filter(quarter %in% input$quarter) %>%
        dplyr::filter(location %in% input$bin_centre_selection) %>%
        mutate(calculated_weight = case_when(
          input$weight_unit == "calculated_weight_kg" ~ calculated_weight_kg,
          input$weight_unit == "calculated_weight_tonnes" ~ calculated_weight_tonnes))
      
      ##plot_functions
      plot_bar <- function(input_group, xlab) { #for categorical x-axis
        ggplot(data = bar_df_processed_filtered,
               aes(x = input_group, y = calculated_weight, fill = location)) +
          scale_fill_brewer(palette = "Spectral") +
          geom_bar(stat = 'identity', position = "dodge") + 
          labs(x = xlab,
               y = ifelse(input$weight_unit == "calculated_weight_kg",
                          "Weight (kg)",
                          "Weight (Tonnes)"),
               title = "Total weight of General Waste\n collected by Bin Centre") +
          guides(fill = guide_legend(title = "Bin Centre")) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
          geom_text(aes(label = calculated_weight), 
                    vjust = -0.4, 
                    position = position_dodge(0.9), 
                    size = 3.5,
                    fontface = "bold")
      }
      
      plot_bar_v2 <- function(input_group, xlab) { #for continuous x-axis
        ggplot(data = bar_df_processed_filtered,
               aes(x = input_group, y = calculated_weight, fill = location)) +
          scale_fill_brewer(palette = "Spectral") +
          geom_bar(stat = 'identity', position = "dodge") + 
          labs(x = xlab,
               y = ifelse(input$weight_unit == "calculated_weight_kg",
                          "Weight (kg)",
                          "Weight (Tonnes)"),
               title = "Total weight of General Waste\ncollected by Bin Centre") +
          guides(fill = guide_legend(title = "Bin Centre")) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
          geom_text(aes(label = calculated_weight), 
                    vjust = -0.4, 
                    position = position_dodge(0.9), 
                    size = 3.5,
                    fontface = "bold") +
          scale_x_continuous(breaks = min(input_group):max(input_group))
      }
      
      ##cater for user configuration + plot bar chart
      if (input$group == "month_year") {
        xlab = "Month & Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          mutate(month_year = factor(month_year, levels = unique(month_year))) %>%
          group_by(location, month_year, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar(bar_df_processed_filtered$month_year, xlab)
      } else if (input$group == "year") {
        xlab = "Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, year, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar_v2(bar_df_processed_filtered$year, xlab)
      } else if (input$group == "wday") {
        xlab = "Day of Week"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, wday, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar(bar_df_processed_filtered$wday, xlab)
      } else if (input$group == "day") {
        xlab = "Day of Month"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, day, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar_v2(bar_df_processed_filtered$day, xlab)
      } else if (input$group == "month") {
        xlab = "Month of Year"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, month, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar(bar_df_processed_filtered$month, xlab)
      } else if (input$group == "fiscal_year") {
        xlab = "Day of Week"
        bar_df_processed_filtered = 
          bar_df_processed_filtered %>%
          group_by(location, fiscal_year, .drop = TRUE) %>%
          summarise(calculated_weight = sum(calculated_weight), .groups = "keep")
        plot_bar(bar_df_processed_filtered$fiscal_year, xlab)
      } else { #group_by(start_date) selected, smallest granularity, no need to group
        xlab = "Date"
        plot_bar(bar_df_processed_filtered$start_date, xlab)
      } 
    })
  })
  
  output$linePlot <- renderPlot({
    #Refresh data from database
    if (flag == "database"){
      if (counter$countervalue != 0) {
        df_processed = df_refreshed()
      }
    }
    #execute when "Apply Configuration" is clicked
    input$goButton_graph
    isolate({
      ##filters
      line_selected_start_date = date(parse_date_time(input$dates_graph[1], "ymd", tz = "Singapore"))
      line_selected_end_date = date(parse_date_time(input$dates_graph[2], "ymd", tz = "Singapore"))
      line_df_processed_filtered =
        df_processed %>%
        dplyr::filter(between(start_date, line_selected_start_date, line_selected_end_date)) %>%
        dplyr::filter(semester %in% input$semester) %>%
        dplyr::filter(quarter %in% input$quarter) %>%
        dplyr::filter(location %in% input$bin_centre_selection) %>%
        mutate(calculated_weight = case_when(
          input$weight_unit == "calculated_weight_kg" ~ calculated_weight_kg,
          input$weight_unit == "calculated_weight_tonnes" ~ calculated_weight_tonnes))
      
      ##plot_functions
      plot_line <- function(input_group, xlab) {
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste`)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste`),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(ifelse(input$weight_unit == "calculated_weight_kg",
                        min(dashboard_line$`Total Generated Waste`)-200,
                        min(dashboard_line$`Total Generated Waste`)-0.2),
                 ifelse(input$weight_unit == "calculated_weight_kg",
                        max(dashboard_line$`Total Generated Waste`)+100,
                        max(dashboard_line$`Total Generated Waste`)+0.1)))+
          labs(x = xlab,
               y = ifelse(input$weight_unit == "calculated_weight_kg",
                          "Weight (kg)",
                          "Weight (Tonnes)"),
               title = paste0("Total weight of General Waste\ncollected by ", xlab)) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      }
      
      plot_line_v2 <- function(input_group, xlab) { #for continuous x-axis
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste`)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste`),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(ifelse(input$weight_unit == "calculated_weight_kg",
                        min(dashboard_line$`Total Generated Waste`)-200,
                        min(dashboard_line$`Total Generated Waste`)-0.2),
                 ifelse(input$weight_unit == "calculated_weight_kg",
                        max(dashboard_line$`Total Generated Waste`)+100,
                        max(dashboard_line$`Total Generated Waste`)+0.1)))+
          labs(x = xlab,
               y = ifelse(input$weight_unit == "calculated_weight_kg",
                          "Weight (kg)",
                          "Weight (Tonnes)"),
               title = paste0("Total weight of General Waste\ncollected by ", xlab)) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
          scale_x_continuous(breaks = min(input_group):max(input_group))
      }
      
      plot_line_v3 <- function(input_group, xlab) { #for wday and month of year bc categorical variables
        ggplot(data = dashboard_line,
               aes(x = input_group, y = `Total Generated Waste`, group = 1)) +
          geom_line(linewidth = 1.5, color = 'darkgray') +
          geom_point(size = 3, color = 'darkgray') +
          geom_text(aes(label = `Total Generated Waste`),
                    vjust = 2,
                    size = 3.5,
                    fontface = "bold") +
          ylim(c(ifelse(input$weight_unit == "calculated_weight_kg",
                        min(dashboard_line$`Total Generated Waste`)-200,
                        min(dashboard_line$`Total Generated Waste`)-0.2),
                 ifelse(input$weight_unit == "calculated_weight_kg",
                        max(dashboard_line$`Total Generated Waste`)+100,
                        max(dashboard_line$`Total Generated Waste`)+0.1)))+
          labs(x = xlab,
               y = ifelse(input$weight_unit == "calculated_weight_kg",
                          "Weight (kg)",
                          "Weight (Tonnes)"),
               title = paste0("Total weight of General Waste\ncollected by ", xlab)) +
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
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v3(dashboard_line$month_year, xlab)
      } else if (input$group == "year") {
        xlab = "Year"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(year, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v2(dashboard_line$year, xlab)
      } else if (input$group == "wday") {
        xlab = "Day of Week"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(wday = factor(wday, levels = unique(wday))) %>%
          group_by(wday, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v3(dashboard_line$wday, xlab)
      } else if (input$group == "day") {
        xlab = "Day of Month"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(day, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v2(dashboard_line$day, xlab)
      } else if (input$group == "month") {
        xlab = "Month of Year"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(month = factor(month, levels = unique(month))) %>%
          group_by(month, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v3(dashboard_line$month, xlab)
      } else if (input$group == "fiscal_year") {
        xlab = "Fiscal Year"
        dashboard_line =
          line_df_processed_filtered %>%
          mutate(fiscal_year = factor(fiscal_year, levels = unique(fiscal_year))) %>%
          group_by(fiscal_year, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line_v3(dashboard_line$fiscal_year, xlab)
      } else { #group_by(start_date) selected, smallest granularity, no need to group
        xlab = "Date"
        dashboard_line =
          line_df_processed_filtered %>%
          group_by(start_date, .drop = FALSE) %>%
          summarise("Total Generated Waste" = sum(calculated_weight), .groups = 'keep')
        plot_line(dashboard_line$start_date, xlab)
      } 
    })
  })
}
  
#   # Ensure button is clicked once upon initialisation
#   ignoreInit = FALSE, ignoreNULL = FALSE) 
# }

# Run the application 
shinyApp(ui = ui, server = server)
