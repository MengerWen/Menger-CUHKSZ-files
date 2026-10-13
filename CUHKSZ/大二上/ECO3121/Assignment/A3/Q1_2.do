// Load the household data
use "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\aghousehold.dta", clear

// Question 1: Data Preparation
// Keep data for the year 2001 only
keep if year == 2001

// Generate variables for yield and fertilizer application intensity
gen yield = hx95_nh125 / hx95_nh112
gen fertilizer = nh6_3 / hx95_nh112

// Generate binary variable for migration decision
gen migrate = (h95_nh269 > 0)

// Generate natural log transformations for yield and fertilizer application
gen ln_yield = log(yield + 1)
gen ln_fertilizer = log(fertilizer + 1)

// Question 2: Linear Probability Model
// Linear probability model for migration decision on ln_yield
reg migrate ln_yield

// Question 3: Merge with Rainfall Data
// Load the rainfall data and merge by village ID
merge m:1 vl_id using "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\village_rainfall.dta"

// Question 4: Instrumental Variable Relevance
// Generate natural log of average rainfall
gen ln_av_rain = log(av_rain)

// First stage regression to test instrument relevance
reg ln_yield ln_av_rain

// Question 5: Second Stage IV Regression
// Estimate 2nd stage IV point estimate
reg migrate (ln_yield = ln_av_rain)

// Question 6: Replication using ivregress
// Use ivregress 2sls to replicate the results
ivregress 2sls migrate (ln_yield = ln_av_rain)

// Save the dataset for further analysis
save "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\processed_data.dta", replace