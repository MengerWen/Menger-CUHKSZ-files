clear

cap log close

cd"/Users/hanqihang/Stata/FIN3080/HW1"

log using HW1_q12.log,  replace

frame create stock
frame create eps
frame create EDMT
frame create Ass
frame create ROAROE

frame change stock

use Stock.dta

//get the quarter of the date
gen date = date(Trdmnt , "YM")
format date %td
gen year = year(date)
gen month = month(date)
gen quarter =.
replace quarter = 1 if inlist(month, 1, 2, 3)
replace quarter = 2 if inlist(month, 4, 5, 6)
replace quarter = 3 if inlist(month, 7, 8, 9)
replace quarter = 4 if inlist(month, 11, 12, 10)
gen date_yq = string(year) + "-Q" + string(quarter)
save "processed_Stock.dta", replace

frame change eps

use eps.dta
drop if Typrep == "B"

gen date = date(Accper, "YMD")
format date %td
gen year = year(date)
gen month = month(date)
gen quarter =.
replace quarter = 1 if inlist(month, 1, 2, 3)
replace quarter = 2 if inlist(month, 4, 5, 6)
replace quarter = 3 if inlist(month, 7, 8, 9)
replace quarter = 4 if inlist(month, 11, 12, 10)
gen date_yq = string(year) + "-Q" + string(quarter)
save "processed_eps.dta", replace

frame change ROAROE

use ROAROE.dta
drop if Typrep == "B"

gen date = date(Accper, "YMD")
format date %td
gen year = year(date)
gen month = month(date)
gen quarter =.
replace quarter = 1 if inlist(month, 1, 2, 3)
replace quarter = 2 if inlist(month, 4, 5, 6)
replace quarter = 3 if inlist(month, 7, 8, 9)
replace quarter = 4 if inlist(month, 11, 12, 10)
gen date_yq = string(year) + "-Q" + string(quarter)
save "processed_ROAROE.dta", replace

frame change Ass

use AssetLiabilityExpenses.dta
drop if Typrep == "B"

gen date = date(Accper, "YMD")
format date %td
gen year = year(date)
gen month = month(date)
gen quarter =.
replace quarter = 1 if inlist(month, 1, 2, 3)
replace quarter = 2 if inlist(month, 4, 5, 6)
replace quarter = 3 if inlist(month, 7, 8, 9)
replace quarter = 4 if inlist(month, 11, 12, 10)
gen date_yq = string(year) + "-Q" + string(quarter)
save "processed_Ass.dta", replace

frame change stock

//merge the data together
merge m:m Stkcd date_yq using processed_eps
drop _merge
merge m:m Stkcd date_yq using processed_ROAROE
drop _merge
merge m:m Stkcd date_yq using processed_Ass
drop _merge
merge m:1 Stkcd using EDMT
drop _merge
rename B003000000 eps
rename F050201B ROA
rename F050501B ROE
rename A001219000 RDex
rename A001000000 totalAss
rename A002000000 totalLia

//calculate the needed data
gen PE = Mclsprc / eps

gen totalTradableShare = Msmvosd * 1000 / Mclsprc
gen B = (totalAss - totalLia) / totalTradableShare
label variable B "Book Value per Share"
gen PB = Mclsprc / B

gen RAratio = RDex / totalAss
label variable RAratio "RDex / totalAss"

gen date1 = date(Trdmnt, "YM")
gen esdt = date(Estbdt, "YMD")
label variable esdt "days of established date"

gen firm_ages = date1 - esdt 

gen market_type = "Main" if Markettype != 16 & Markettype != 32
replace market_type = "GEM" if Markettype == 16 | Markettype == 32

sort Stkcd Trdmnt
save "Result", replace

//Main//
estpost summarize Mretwd PE PB ROA ROE RAratio firm_ages if market_type == "Main", detail
eststo ss_main
esttab ss_main, cell(b(fmt(2)) "count mean p25 p50 p75 sd" ) title("Descriptive Statistics for Main Market")

//GEM//
estpost summarize Mretwd PE PB ROA ROE RAratio firm_ages if market_type == "GEM", detail
eststo ss_main
esttab ss_main, cell(b(fmt(2)) "count mean p25 p50 p75 sd") title("Descriptive Statistics for GEM Market")


//question 2//
collapse (median) PE, by(market_type year)
twoway (line PE year if market_type == "GEM", lcolor(blue) lwidth(medium) lpattern(solid)) (line PE year if market_type != "GEM", lcolor(red) lwidth(medium) lpattern(dash)), title("Median P/E Ratio by Market Type") legend(label(1 "GEM") label(2 "Main Markets")) xlabel(, angle(45)) ylabel(, grid)


log close

