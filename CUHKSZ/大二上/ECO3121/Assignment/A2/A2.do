clear

use "D:\MG\！CUHKSZ\~！大二上\ECO3121\Assignment\A2\inpresdata.dta", clear

//Question 1

//(1)
//model 1
reg wage yeduc nin

//(2)
//model 2
reg nin yeduc
reg wage yeduc

//(3)
// Check the correlation between nin & yeduc
correlate nin yeduc

//(4)
//model 3
reg wage yeduc nin en71

//(6)
//Find the joint significance of nin and en71
test nin en71

//---
//Question 2

//(1)
gen high_wage = . 
replace high_wage = 1 if wage >= 192000 & !missing(wage)
replace high_wage = 0 if wage < 192000

//linear model
reg high_wage yeduc nin

//probit model
probit high_wage yeduc nin

//---
//Question 3
//(1)
//Generate high_educ
gen high_level = . 
replace high_level = 1 if yeduc >= 9 & !missing(wage)
replace high_level = 0 if yeduc < 9

//Show the relationship between wage & high level
reg wage high_level

//(2)
//Generate urbanhigh_level
gen interaction = urban * high_level

//Regress in the interaction term
reg wage high_level urban interaction

//Check if the urban will affect the slope of high_level
reg urban interaction