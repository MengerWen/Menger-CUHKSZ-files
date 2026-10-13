* Set the working directory where the datasets are located
cd "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3"

* Step 1: Load and clean the household data
use aghousehold.dta, clear

* Limit the sample to households in the year 2001
keep if year == 2001

* Generate yield (output per unit of land)
gen yield = hx95_nh125 / hx95_nh112

* Generate fertilizer application per unit of land
gen fertilizer = nh6_3 / hx95_nh112

* Generate a binary variable for migration decision
gen migrate = (h95_nh269 > 0)

* Generate the natural log of yield and fertilizer
gen ln_yield = log(yield + 1)
gen ln_fertilizer = log(fertilizer + 1)

* Save the cleaned household data
save cleaned_aghousehold.dta, replace

* Step 2: Load and clean the rainfall data
use village_rainfall.dta, clear

* Save the cleaned rainfall data
save cleaned_village_rainfall.dta, replace

* Step 3: Merge the two datasets on village ID (vl_id)
use cleaned_aghousehold.dta, clear
merge m:1 vl_id using cleaned_village_rainfall.dta

* Step 4: First stage regression for IV relevance
* Generate the natural log of average rainfall
gen ln_av_rain = log(av_rain)

* Run the first stage regression: ln_yield on ln_av_rain
reg ln_yield ln_av_rain

* Output interpretation
display "First Stage Regression Model:"
display "ln_yield = β0 + β1*ln_av_rain + ε"
display "Interpretation: The coefficient of ln_av_rain indicates the effect of average rainfall on yield. Statistical significance is determined by the p-value (p < 0.05 indicates significance)."

* Step 5: Second stage IV regression (manually, two-step approach)
* First step: Predict ln_yield using ln_av_rain
predict ln_yield_hat, xb

* Second step: Use ln_yield_hat as an instrument for ln_yield in the migration regression
reg migrate ln_yield_hat

* Output interpretation for the second stage regression
display "Second Stage Regression Model:"
display "migrate = β0 + β1*ln_yield_hat + ε"
display "Interpretation: The coefficient of ln_yield_hat indicates the causal effect of agricultural yield on household migration decisions."

* Step 6: Use ivregress 2sls to directly estimate the IV model
ivregress 2sls migrate (ln_yield = ln_av_rain)

* Compare results with the manual 2-stage regression
display "Comparison of coefficients and standard errors:"
display "Check if the results from 'ivregress 2sls' match those from the manual 2-stage estimation."

* Step 7: Discussion of differences in IV and OLS results
display "Comparison with OLS (Question 2):"
display "The IV estimate captures the causal effect of yield on migration decisions, addressing endogeneity. Differences between OLS and IV estimates highlight potential bias in OLS."
display "Comparison with Question 3:"
display "The difference between IV and OLS estimates aligns with expectations if biases (e.g., measurement error or reverse causality) were present in the OLS model."