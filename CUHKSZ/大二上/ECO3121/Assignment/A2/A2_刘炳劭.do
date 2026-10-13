use "C:\Users\Lenovo\Desktop\大二上\eco3121\ass2\inpresdata.dta", clear
display "Q1"

display "1."
regress wage yeduc nin
regress nin yeduc

display "2."
regress wage yeduc

display "3."
correlate yeduc nin

display "4."
regress wage yeduc nin en71

display "6."
test nin en71

display "Q2"

display "1."
generate high_wage = . 
replace high_wage = 1 if wage >= 192000 & !missing(wage)
replace high_wage = 0 if wage < 192000

regress high_wage yeduc nin
probit high_wage yeduc nin
logit high_wage yeduc nin

display "Q3"

display "1."
generate high_level = . 
replace high_level = 1 if yeduc >= 9 & !missing(wage)
replace high_level = 0 if yeduc < 9

regress wage high_level

display "2."
gen interaction = urban * high_level
regress wage high_level urban interaction