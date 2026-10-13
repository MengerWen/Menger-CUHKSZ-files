* Set the directory for the datasets
cd "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3"

* Load the household data
use aghousehold.dta, clear

* Question 1: Limit the sample to households in the year 2001
keep if year == 2001

* Generate the variables 'yield' and 'fertilizer'
gen yield = hx95_nh125 / hx95_nh112
gen fertilizer = nh6_3 / hx95_nh112

* Generate a binary variable for migration decision
gen migrate = (h95_nh269 > 0)

* Generate natural log variables for yield and fertilizer
* (log + 1 transformation to smooth zero values)
gen ln_yield = log(yield + 1)
gen ln_fertilizer = log(fertilizer + 1)

* Save the cleaned household data for merging
save cleaned_aghousehold.dta, replace

* Load the rainfall data
use village_rainfall.dta, clear

* Save the cleaned rainfall data
save cleaned_village_rainfall.dta, replace

* Merge the two datasets on village ID (vl_id)
use cleaned_aghousehold.dta, clear
merge m:1 vl_id using cleaned_village_rainfall.dta

* Question 2: Run a linear probability model of household migration decision on ln_yield
reg migrate ln_yield

* Question 4: First stage regression for IV relevance
* Generate the natural log of average rainfall
gen ln_av_rain = log(av_rain)

* Make sure the 2SLS is operate on the same, valid dataset
drop if missing(ln_yield, ln_av_rain, migrate)

* Run the first stage regression: ln_yield on ln_av_rain
reg ln_yield ln_av_rain

* Question 5: Second stage IV regression (manually, two-step approach)
* First step: Predict ln_yield using ln_av_rain
predict ln_yield_hat, xb

* Check whether all the data are valid
summ ln_yield_hat
list if missing(ln_yield_hat)

* Second step: Use ln_yield_hat as an instrument for ln_yield in the migration regression
reg migrate ln_yield_hat

* Caompare to the first & second stage of the 2SLS
count if !missing(ln_yield, ln_av_rain, migrate)
count if !missing(ln_yield_hat, migrate)

* Question 6: Use ivregress 2sls to directly estimate the IV model
ivregress 2sls migrate (ln_yield = ln_av_rain)