* Set the working directory where your file is located
cd "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A4"

* Import the CSV file into Stata
import delimited DID_Example.csv, clear

* Replace "NA" with missing values
foreach var in ID Treatment Group Empl C.Owned Hours.Opening Soda Fries Chain SouthJ CentralJ NorthJ PA1 PA2 Shore {
    replace `var' = . if `var' == "NA"
}

* Convert string variables to numeric where applicable
foreach var in ID Treatment Group Empl C.Owned Hours.Opening Soda Fries Chain SouthJ CentralJ NorthJ PA1 PA2 Shore {
    destring `var', replace
}

* Generate the interaction term for difference-in-differences
gen DID = Treatment * Group

* Run the regression model with control variables
regress Empl Treatment Group DID C.Owned Hours.Opening Soda Fries

* Display regression results
outreg2 using DID_Regression_Results.doc, replace ctitle("Difference-in-Differences Regression")