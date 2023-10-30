#install.packages("ggplot2")
#install.packages("gridExtra")

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

#1. Tabular data

##Summarise by location
location_df = read.csv("location_database.csv")

dashboard_tabular = 
  df_processed %>% 
  group_by(location, .drop = FALSE) %>% 
  summarise("Total Weight in kg" = sum(weight_kg), .groups = 'keep')

dashboard_tabular_with_loc = 
  location_df %>%
  inner_join(dashboard_tabular, by = c("bin_centre" = "location"))

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
#selected_end_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
bar_selected_end_date = Sys.Date()
bar_df_processed_filtered = 
  df_processed %>%
  filter(between(start_date, bar_selected_start_date, bar_selected_end_date))

##plot
ggplot(data = bar_df_processed_filtered,
       aes(x = start_date, y = weight_kg, fill = location)) +
  geom_bar(stat = 'identity', position = "dodge") + 
  labs(x = "Date",
       y = "Weight (kg)",
       title = "Total weight of General Waste collected by Bin Centre") +
  guides(fill = guide_legend(title = "Bin Centre")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  geom_text(aes(label = weight_kg), 
            vjust = -0.4, 
            position = position_dodge(0.9), 
            size = 3.5,
            fontface = "bold")

#3. Line Plot
##filters
line_selected_start_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
#selected_end_date = date(parse_date_time("16-10-2023", "dmy", tz = "Singapore"))
line_selected_end_date = Sys.Date()
line_df_processed_filtered = 
  df_processed %>%
  filter(between(start_date, line_selected_start_date, line_selected_end_date))

##Summarise by date
dashboard_line = 
  line_df_processed_filtered %>% 
  group_by(start_date, .drop = FALSE) %>% 
  summarise("Total Weight in kg" = sum(weight_kg), .groups = 'keep')

##plot
ggplot(data = dashboard_line, 
       aes(x = start_date, y = `Total Weight in kg`)) + 
  geom_line(linewidth = 1.5, color = 'gray') +  
  geom_point(size = 3, color = 'gray') +
  geom_text(aes(label = `Total Weight in kg`), 
            #position = position_jitter(width = 0, height = 300), 
            vjust = 2,
            size = 3.5,
            fontface = "bold") +
  ylim(c(min(dashboard_line$`Total Weight in kg`-200),max(dashboard_line$`Total Weight in kg`)))+
  labs(x = "Date",
       y = "Weight (kg)",
       title = "Total weight of General Waste collected by Date") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
 
  
