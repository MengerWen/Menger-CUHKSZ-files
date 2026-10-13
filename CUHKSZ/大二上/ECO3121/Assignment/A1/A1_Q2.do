* Load the data from the provided path
use "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A1\inpresdata.dta", clear

* Step 2: Running the Simple Linear Regression: wage on years of education
reg wage yeduc

* Step 4: Compute fitted values and residuals
predict fitted_wage
predict residuals, residuals

* Verify if residuals sum to zero
summ residuals

* Step 5: Predicted wage for 10 years of education
display _b[_cons] + _b[yeduc]*10

* Step 7: Exploring alternative functional forms

* Log-Linear Model: log(wage) on education
gen log_wage = log(wage)
reg log_wage yeduc

* Linear-Log Model: wage on log(education)
gen log_educ = log(yeduc)
reg wage log_educ

* Log-Log Model: log(wage) on log(education)
reg log_wage log_educ

* Step 8: Plot residuals vs fitted values to check for assumption SLR.4
rvfplot
