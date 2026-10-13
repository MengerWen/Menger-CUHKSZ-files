clear

cap log close

cd"/Users/hanqihang/Stata/FIN3080/HW1"

log using HW1_q3.log,  replace

import delimited "problem3.csv", clear

gen time = date(enddate, "YMD")

gen year = year(time)



tsset symbol year

gen revenue_growth_pct = (totalrevenue - L.totalrevenue) / L.totalrevenue
drop if missing(revenue_growth_pct)

keep if year >= 2011 & year <= 2020

by symbol (year), sort: gen year_count = _N

keep if year_count == 10

drop year_count

sort year



egen median_roe = median(roec), by(year)

egen median_growth = median(revenue_growth_pct), by(year)

gen above_median_roe = (roec > median_roe)
gen above_median_growth = (revenue_growth_pct > median_growth)

sort symbol

gen consistent_roe =.

sort year symbol

bysort symbol (year) : replace consistent_roe = 1 if (L.consistent_roe == 1. | L.consistent_roe ==.) & (year == 2011 | (above_median_roe == 1 & L.above_median_roe == 1))
bysort symbol (year) : replace consistent_roe = 0 if L.consistent_roe == 0 |(above_median_roe == 0 | L.above_median_roe == 0)

gen consistent_growth =.
bysort symbol (year) : replace consistent_growth = 1 if (L.consistent_growth == 1 | L.consistent_growth ==.) & (year == 2011 | (above_median_growth == 1 & L.above_median_growth == 1))
bysort symbol (year) : replace consistent_growth = 0 if L.consistent_growth == 0 | (above_median_growth == 0 | L.above_median_growth == 0)

sort year

egen consistent_roe_pct = mean(consistent_roe), by(year)
egen consistent_growth_pct = mean(consistent_growth), by(year)

sort year

twoway (line consistent_roe_pct year, lcolor(blue) lwidth(medium) lpattern(solid)) (line consistent_growth_pct year, lcolor(red) lwidth(medium) lpattern(dash)), title("Percentage of Firms Consistently Maintaining Above-Median ROE and Revenue Growth") legend(label(1 "Consistent ROE") label(2 "Consistent Revenue Growth")) xlabel(2011(1)2020) ylabel(, grid)













