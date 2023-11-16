# README

## General Information
This directory holds all files that is required for the Shiny R Dashboard solution, which is programmed in R language. Prominent libraries, other than `shiny`, used in this dashboard are `dplyr`, `ggplot`, DataTables (`DT`), and `RMySQL` (for communicating with MySQL Database). *Full list of libraries used can be found in lines `1` to `10` in `./app/app.R`.

### Dashboard Key Features
![Dashboard Screenshot](./images/filename.png "Dashboard Screenshot")*Screenshot of Dashboard*
1. **Bar Chart**: Users are able to visualise data on a bar chart for comparisons of waste weight between particular bin centres across dates. Data is pulled from connected MySQL Database, and can be refreshed.
2. **Line Plot**: Users can also visualise data on a line plot for comparisons of waste weight across dates (time series analysis). Data is also pulled from connected MySQL Database, and can be refreshed.
3. **DataTable**: For users who prefer clear numerical and definitive values, they can refer to the DataTable which displays waste weight by each bin centre. Users can also download the data in CSV or Excel file formats for further data manipulation
4. **Dashboard Configurations**: Users can configure and customise the Dashboard through the Side Panel. Some of the possible configurations include: Date Range Selection, Group by (day, month, year, ...), Quarter/Semester Date Range Filters, Weight Units (kg/tonnes), Bin Centre Selection. 

### Related files
1. `./images`: Folder containing screenshot of Dashboard
2. `./app`: Folder containing files that is required for Dashboard to work
    1. `./app/app.R`: Main R script to program the Dashboard
    2. `./app/dashboard_dockerfile`: Dockerfile for building the dashboard image and container
    3. `./app/location_database.csv`: CSV file consisting of Zones and Precincts data for each bin centre
    4. `./app/test_data.csv`: CSV file used for debugging purposes
3. `./test_data_extra.csv`: Larger CSV file used for debugging purposes, most of which are dummy data
4. `./plot_templates.R`: Templates for bar chart and line plots programmed in R 

