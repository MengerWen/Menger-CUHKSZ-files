* Set the working directory where your file is located
cd "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A4"

* Import the CSV file into Stata
import delimited DID_Example.csv, clear

* Replace "NA" with missing values
replace empl = "" if empl == "NA"
replace soda = "" if soda == "NA"
replace fries = "" if fries == "NA"

* Convert string variables to numeric where applicable
destring empl, replace
destring soda, replace
destring fries, replace

* Generate the interaction term for difference-in-differences
gen treatmentgroup = treatment * group

* Run the regression model with control variables
reg empl treatment group treatmentgroup fries