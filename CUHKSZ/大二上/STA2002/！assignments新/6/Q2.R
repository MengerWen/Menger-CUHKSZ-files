# Step 1: Define sample sizes and numbers with anemia
n1 <- 2100  # Sample size for African women
y1 <- 840   # Number with nutritional anemia in Africa
n2 <- 1900  # Sample size for women in the Americas
y2 <- 323   # Number with nutritional anemia in the Americas

# Calculate proportions
p1 <- y1 / n1  # Proportion for Africa
p2 <- y2 / n2  # Proportion for Americas

# Output the proportions
cat("Proportion of women with anemia in Africa (p1):", p1, "\n")
cat("Proportion of women with anemia in Americas (p2):", p2, "\n\n")

# Step 2: Calculate the standard error
SE <- sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

# Output the standard error
cat("Standard Error (SE):", SE, "\n\n")

# Step 3: Calculate the confidence interval
Z <- qnorm(0.95)  # Z-value for 90% confidence level
difference <- p1 - p2  # Difference in proportions
margin_of_error <- Z * SE  # Margin of error

# Calculate the confidence interval
CI_lower <- difference - margin_of_error
CI_upper <- difference + margin_of_error

# Output the results
cat("Difference in proportions (p1 - p2):", difference, "\n")
cat("90% Confidence Interval for the difference:(", CI_lower, ",", CI_upper, ")\n")
