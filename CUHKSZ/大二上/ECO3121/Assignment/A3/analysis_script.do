// Load the household data
use "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\aghousehold.dta", clear

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

// Load the rainfall data
merge m:1 vl_id using "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\village_rainfall.dta"

// Linear probability model for migration decision on ln_yield
reg migrate ln_yield

// Save the dataset for further analysis
save "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A3\processed_data.dta", replace