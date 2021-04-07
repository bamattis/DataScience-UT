# Title: C3T4 Market Basket Analysis
# Author: Brian Mattis
# Date: 3/31/2021

# Project name: Course 3, Task 4

###############
# Project Notes
###############

# Summarize project: Perform a market basket analysis using the Electronidex  
# Transaction history and analyze for interesting product relationships and 
# if Electronidex would be a good acquisition for Blackwell Electronics

################
# Load packages
################

library(readr)
library(arules)
library(arulesViz)
library(lazyeval)

##############
# Import data 
##############

dfOOB <- read.transactions("data/ElectronidexTransactions2017.csv", format="basket",
                           header=FALSE, sep=",",cols=NULL, rm.duplicates=TRUE, skip=0)

# https://www.rdocumentation.org/packages/arules/versions/1.6-7/topics/read.transactions
###############
# Data Cleaning
###############

#None needed

################
# Evaluate data
################
?inspect
inspect(dfOOB)
#just lists out the rows in the console... not interactive
inspect(dfOOB, fun=NULL)
length(dfOOB) #number of rows
# [1] 9835
size(dfOOB) #number of items per transaction
LIST(dfOOB) #transactions by conversion
itemLabels(dfOOB)

summary(dfOOB)
#transactions as itemMatrix in sparse format with
#9835 rows (elements/itemsets/transactions) and
#125 columns (items) and a density of 0.03506172 
#
#most frequent items:
#  iMac                HP Laptop        CYBERPOWER Gamer Desktop      Apple Earpods 
#  2519                 1909                     1809                     1715 
#Apple MacBook Air       (Other) 
#1530                    33622 

#element (itemset/transaction) length distribution:
#  sizes
#0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21 
#2 2163 1647 1294 1021  856  646  540  439  353  247  171  119   77   72   56   41   26   20   10   10   10 
#22   23   25   26   27   29   30 
#5    3    1    1    3    1    1 

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.000   2.000   3.000   4.383   6.000  30.000 

#####################
# EDA/Visualizations
#####################

itemFrequencyPlot(dfOOB, type="absolute", topN=20)
itemFrequencyPlot(dfOOB, type="absolute", topN=20, col="lightblue", cex.names=0.8)
itemFrequencyPlot(dfOOB, type="absolute", topN=20, col="lightblue", horiz=TRUE, cex.names=0.8)

image(dfOOB)
#select just the first 50 transactions
image(head(dfOOB, 50))
#50 random transactions
image(sample(dfOOB, 50))
#look at the items for transactions with > 20 items in it
image(subset(dfOOB, size(dfOOB)>20))
#look at times for transactions with only 1 item
image(head(subset(dfOOB, size(dfOOB)==1),25))
#look at the items for transactions with > 15 items in it
image(head(subset(dfOOB, size(dfOOB)>15),25))
#sort by transaction size, plot the ones with the most items
image(head(dfOOB[order(size(dfOOB), decreasing=TRUE)],50))
      
#####################
# Modelling - Apriori Algorithm
#####################

RulesDex <- apriori(dfOOB, parameter=list(supp=0.02, conf=0.5))
# 1 rule generated: 
#lhs                                    rhs    support  confidence coverage  lift    
#{HP Laptop,Lenovo Desktop Computer} => {iMac} 0.02308   0.5      0.0461     1.952
#count: 227.  That seems like a lot of instances.  Maybe reduce supp further
inspect(RulesDex)
#minlen=2 is used to prevent empty lhs creating a rule {}=>{iMac}
#more likely if a given item happens in a majority of transactions
RulesDex <- apriori(dfOOB, parameter=list(supp=0.012, conf=0.5, minlen=2))
inspect(RulesDex)
#11 rules
#support (0.12 - 0.23)
#confidence (0.50 - 0.55)
#lift (1.952 - 2.96)
#thoughts: counts are still pretty high, may be able to relax support to get better confidence/lift

RulesDex <- apriori(dfOOB, parameter=list(supp=0.006, conf=0.6, minlen=2))
inspect(RulesDex)
#17 rules
#support (0.0062 - 0.0107)
#confidence (.601 - .607)
#lift (2.34 - 3.415)
#thoughts - it seems to have paid off

#note - to make rules with a specific RHS, titanic example shows:
# rules <- apriori(titanic.raw, parameter = list(minlen=2, supp=0.005, conf=0.8),
#+ appearance = list(rhs=c("Survived=No", "Survived=Yes"),
#since monitors are associated with iMac/HP Laptop, let's look at it the other way around:
RulesMonitor <- apriori(dfOOB, parameter=list(supp=0.002, conf=0.5, minlen=2), 
                        appearance = list(rhs=c("ViewSonic Monitor","ASUS Monitor")))
inspect(RulesMonitor)
#26 rules
#wow - some really high lifts here if we're willing to relax support this far.

#####################
# Model Evaluation
#####################

summary(RulesDex)

#  support           confidence        coverage             lift           count       
#Min.   :0.006202   Min.   :0.6000   Min.   :0.009354   Min.   :2.343   Min.   : 61.00  
#1st Qu.:0.006406   1st Qu.:0.6224   1st Qu.:0.009964   1st Qu.:2.466   1st Qu.: 63.00  
#Median :0.006914   Median :0.6327   Median :0.010778   Median :2.510   Median : 68.00  
#Mean   :0.007614   Mean   :0.6351   Mean   :0.012052   Mean   :2.710   Mean   : 74.88  
#3rd Qu.:0.008439   3rd Qu.:0.6429   3rd Qu.:0.014032   3rd Qu.:3.103   3rd Qu.: 83.00  
#Max.   :0.010778   Max.   :0.6939   Max.   :0.017895   Max.   :3.416   Max.   :106.00  

inspect(head(RulesDex,3))
#to sort the rules by Lift
RulesLift <- sort(RulesDex, by="lift")
inspect(RulesLift)
#{HP Laptop} rules have the highest Lifts -> most important

#look at the top5 lift ones
inspect(head(RulesLift,5))

#look at only the rules with Dell Desktops involved
DellRules <- subset(RulesDex, items %in% "Dell Desktop")
inspect(DellRules)
# in 5 of 17 rules

#look at only HP laptop rules (pulls LHS and RHS)
HPLaptopRules <- subset(RulesDex, items %in% "HP Laptop")
inspect(HPLaptopRules)
#HP is on both LHS and RHS
iMacRules <- subset(RulesDex, items %in% "iMac")
inspect(iMacRules)
#iMac is in 14 of 17 rules (all but 2 on RHS)

#redundant rules?
is.redundant(RulesDex)
# FALSE on all
#if there were, we do something like: 
# rules.pruned <- rules.sorted[!redundant]

#####################
# Rule Visualization
#####################

plot(RulesDex, cex=1.3)
plot(RulesLift[1:6], method="graph", control=list(type="items"), cex=0.8)
#ViewSonic Monitor seems highly linked to both iMac and HP Laptop rules
# - in 9 of 17 rules on LHS
plot(RulesDex[1:17], method="graph", control=list(type="items"))
#interactive plot
plot(RulesDex, engine="plotly")
plot(RulesDex, method="grouped")
