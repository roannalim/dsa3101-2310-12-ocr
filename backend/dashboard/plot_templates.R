#install.packages("ggplot2")
#install.packages("gridExtra")
setwd("C:/Users/wenji/Desktop/NUS Academics/NUS Y3S1/DSA3101/dsa3101-2310-12-ocr/backend/dashboard")

library(ggplot2)
library(ggrepel)
library(DT)

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
  mutate(year = year(start_date)) %>%
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

#1. Tabular data
##filters
table_selected_start_date = date(parse_date_time("01-10-2023", "dmy", tz = "Singapore"))
#table_selected_end_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
table_selected_end_date = Sys.Date()
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
###DT tips: https://clarewest.github.io/blog/post/making-tables-shiny/
DT::datatable(dashboard_tabular_with_loc,
          options = list(paging = TRUE,
                         scrollX = TRUE,
                         scrollY = TRUE,
                         autoWidth = TRUE,
                         buttons = c('csv', 'excel'),
                         dom = 'Bfrtip'),
          extensions = 'Buttons',
          selection = 'multiple',
          filter = 'bottom',
          rownames = FALSE
          )

#2. Bar Charts
##filters
bar_selected_start_date = date(parse_date_time("01-10-2023", "dmy", tz = "Singapore"))
#bar_selected_end_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
bar_selected_end_date = Sys.Date()
bar_df_processed_filtered = 
  df_processed %>%
  filter(between(start_date, bar_selected_start_date, bar_selected_end_date)) %>%
  group_by(location, month, .drop = TRUE) %>%
  summarise(calculated_weight_kg = sum(calculated_weight_kg), .groups = "keep") %>%
  filter(semester %in% list("s1","v1","s2","v2"))

##plot
ggplot(data = bar_df_processed_filtered,
       aes(x = month, y = calculated_weight_kg, fill = location)) +
  geom_bar(stat = 'identity', position = "dodge") + 
  labs(x = "Date",
       y = "Weight (kg)",
       title = "Total weight of General Waste collected by Bin Centre") +
  guides(fill = guide_legend(title = "Bin Centre")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  geom_text(aes(label = calculated_weight_kg), 
            vjust = -0.4, 
            position = position_dodge(0.9), 
            size = 3.5,
            fontface = "bold")

#3. Line Plot
##filters
line_selected_start_date = date(parse_date_time("01-10-2023", "dmy", tz = "Singapore"))
#line_selected_end_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
line_selected_end_date = Sys.Date()
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
  geom_line(linewidth = 1.5, color = 'gray') +  
  geom_point(size = 3, color = 'gray') +
  geom_text(aes(label = `Total Generated Waste (kg)`), 
            #position = position_jitter(width = 0, height = 300), 
            vjust = 2,
            size = 3.5,
            fontface = "bold") +
  ylim(c(min(dashboard_line$`Total Generated Waste (kg)`-200),max(dashboard_line$`Total Generated Waste (kg)`)))+
  labs(x = "Date",
       y = "Weight (kg)",
       title = "Total weight of General Waste collected by Date") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
 
  
