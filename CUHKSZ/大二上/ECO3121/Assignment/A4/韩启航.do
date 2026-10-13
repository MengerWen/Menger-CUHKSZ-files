clear

cap log close

log using ps4.log,  replace

cd"/Users/hanqihang/Stata/PS4"

import delimited "/Users/hanqihang/Stata/PS4/DID_Example.csv"

replace empl = "" if empl == "NA"
replace soda = "" if soda == "NA"
replace fries = "" if fries == "NA"

destring empl, replace
destring soda, replace
destring fries, replace

gen treatmentgroup = treatment * group

reg empl treatment group treatmentgroup fries

log close




