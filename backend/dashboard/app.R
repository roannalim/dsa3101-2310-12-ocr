library(shiny)

test_data = read.csv("test_data.csv")

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("What-a-Waste Dashboard"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
                  "Number of bins:",
                  min = 1,
                  max = 50,
                  value = 30),
      textInput("colour", "Colour:", value="red"),
      actionButton("goButton", "Go!")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$distPlot <- renderPlot({
    # generate bins based on input$bins from ui.R
    input$goButton
    isolate({
      x    <- faithful[, 2]
      bins <- seq(min(x), max(x), length.out = input$bins + 1)
      
      # draw the histogram with the specified number of bins
      hist(x, breaks = bins, col = input$colour, border = 'white',
           xlab = 'Waiting time to next eruption (in mins)',
           main = 'Histogram of waiting times')
    } )
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
