#!/usr/bin/env Rscript

# R script to perform Fisher's Exact Test for contingency tables bigger than 2x2

library("stats")

run_fisher_test <- function(input_csv, output_csv) {
  
  # Read the CSV file into a dataframe
  contingency_table <- read.csv(input_csv, header = TRUE)
  
  # Convert the dataframe to a matrix
  contingency_matrix <- as.matrix(contingency_table)
  
  # Perform Fisher's Exact Test
  fisher_test_result <- fisher.test(contingency_matrix)
  
  # Extract p-value
  first_col = c("p-value")
  second_col= c(fisher_test_result$p.value)
  result <- data.frame(first_col, second_col)
  
  # Write the result to a CSV file
  write.csv(result, output_csv, row.names = FALSE)
}

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Run the function with provided arguments
run_fisher_test(args[1], args[2])
