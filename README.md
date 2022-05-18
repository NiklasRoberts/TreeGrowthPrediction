# AI-Final-Project
Using applied data science techniques to analyze a Forest Ecology dataset measuring bark infections over several years.


Years:
1982
1985
1990

Table columns:
- Tree num
- D82
- D85
- D90
- Nutrients at 1982
- Nutrients at 1985


Tree growth per year:

From 1982 -> 1985
- tree num
- (year)
- (D increase/dprev) / year * 100
- nutrients at 1982

From 1985 -> 1990
- tree num
- (year)
-(D increase/dprev) / year * 100
- nutrients at 1985



Assumptions:
- Trees growth rate is not affected by the trees age
- Foliar nutrient levels are measured at the beginning of the growing time period
- Foliar nutrients remain constant over growth period
- Foliar nutrients for a plot is the same for all trees within that plot
- The same tree may have growth/yr calculated from it twice (once from 82-85, once from 85-90)



Different Models:

- Growth per year bins (2 approaches):
    - Bin by fixed range (0-.2, .2-.4, .4-.6, .6+)
    - Bin by equal record distribution

- Regression
    - one output node, coninuous range