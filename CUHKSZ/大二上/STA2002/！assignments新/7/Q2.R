# Set the significance level and degrees of freedom
alpha <- 0.05
df <- 8

# Calculate the critical t-value
t_critical <- qt(1 - alpha, df)

# Display the result
print(t_critical)