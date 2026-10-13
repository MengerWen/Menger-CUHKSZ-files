* Step 1: Load and Prepare Data
clear
use aghousehold.dta, clear  * Load the household dataset
merge 1:1 vl_id using village_rainfall.dta  * Merge with the village rainfall dataset

* Generate variables for yield and fertilizer
gen yield = hx95_h125 / hx95_nh112
gen fertilizer = nh6_3 / hx95_nh112

* Generate binary migration variable
gen migration = (h95_nh269 > 0)

* Log-transform yield and fertilizer variables
gen log_yield = log(yield + 1)
gen log_fertilizer = log(fertilizer + 1)

* Step 2: Run Linear Probability Model (LPM)
reg migration log_yield
di "Interpret the coefficient of log_yield as the change in probability of migration."

* Step 3: Bias Discussion
di "Discuss potential biases in the LPM and their directions (upwards or downwards)."

* Step 4: First Stage Regression (Instrument Relevance)
gen log_av_rain = log(av_rain + 1)  * Log of average rainfall
reg log_yield log_av_rain  * First stage regression
di "Interpret the coefficient of log_av_rain for instrument relevance."

* Step 5: Second Stage Regression (IV Estimation)
predict log_yield_hat if e(sample)  * Obtain fitted values from first stage
reg migration log_yield_hat  * Second stage regression
di "Interpret the coefficient of log_yield_hat as the effect of fitted yield on migration probability."

* Step 6a: Run IV Regression using ivregress 2sls
ivregress 2sls migration (log_yield = log_av_rain), robust
di "Interpret the coefficient and standard errors from ivregress 2sls."

* Step 6b: Comparison of LPM and IV estimates
di "Discuss the differences between LPM and IV estimates and whether it aligns with expectations based on bias discussion."