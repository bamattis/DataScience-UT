
# DataScience Course 4, Task 1.

# Goal: 
#   Task1: Load and evaulate Data
#   Task2: Data visuallization and regression modelling


# Business Goal: Investigate usage patterns of Summer 2008 for irregularities
# - provide the client with five energy savings suggestions based on insights 
#   that you glean from your analysis


################################
## Install and load packages
################################

# Standard Libraries:
library(RMySQL)
library(tidyr)
library(dplyr)
library(lubridate) #helps with date/time
library(forecast)
library(TTR)
library(plotly)
library(vioplot)
library(ggplot2)
library(ggfortify)
library(forecast)
library(TTR)

###############
# Load dataset 
###############

## Create a database connection 
con = dbConnect(MySQL(), user='deepAnalytics', 
                password='Sqltask1234!', dbname='dataanalytics2018', 
                host='data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com')

## List the tables contained in the database 
dbListTables(con)
#[1] "iris" "yr_2006" "yr_2007" "yr_2008" "yr_2009" "yr_2010"

#### Task Data  ####
dbListFields(con,'yr_2006')
yr_2006 <- dbGetQuery(con, "SELECT Date, Time, Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2006")
yr_2007 <- dbGetQuery(con, "SELECT Date, Time, Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2007")
yr_2008 <- dbGetQuery(con, "SELECT Date, Time, Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2008")
yr_2009 <- dbGetQuery(con, "SELECT Date, Time, Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2009")
yr_2010 <- dbGetQuery(con, "SELECT Date, Time, Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2010")

str(yr_2007)  #date and time are as 'chr'.  Will need to fix that.
summary(yr_2007)
#submeter_3 has the largest average usage for all years
head(yr_2006)  #starts dec 16th
tail(yr_2006)  #ends dec 31st
tail(yr_2010) #ends nov 26th
#our data's date range is 12/16/2006 - 11/26/2010

## Combine tables into one dataframe using dplyr
#   create a primary data frame that ONLY includes
#   the data frames that span an entire year (2007/8/9)
df <- bind_rows(yr_2007, yr_2008, yr_2009)
head(df)
tail(df)
#yes, dates appear to be in decending order
summary(df)
str(df)


#############
# Preprocess
#############

## Combine Date and Time attribute values in a new attribute column
df <-cbind(df,paste(df$Date,df$Time), stringsAsFactors=FALSE)

## Give the new attribute in the 6th column a header name 
colnames(df)[7] <-"DateTime"

## Move the DateTime attribute within the dataset to be first
df <- df[,c(ncol(df), 1:(ncol(df)-1))]
head(df)
str(df)
## Convert DateTime from character to POSIXct 
df$DateTime <- as.POSIXct(df$DateTime,
                          format = "%Y-%m-%d %H:%M:%S",
                          tz = "Europe/Paris")
#Note: important to set tz here rather than adjust the attribute later with
#  attr(df$DateTime, "tzone") <- "Europe/Paris"
#because during as.POSIXct it sets on where data is captured, then attr just does a shift vs GMT

head(df)
tz(df$DateTime) # "Europe/Paris"

# convert Date to as.Date
df$Date <- as.Date(df$Date, "%Y-%m-%d")
str(df)
#'data.frame':	1569894 obs. of  15 variables:
#  $ DateTime      : POSIXct, format: "2007-01-01 00:00:00" "2007-01-01 00:01:00" "2007-01-01 00:02:00" "2007-01-01 00:03:00" ...
#  $ Date          : Date, format: "2007-01-01" "2007-01-01" "2007-01-01" "2007-01-01" ...
#  $ Time          : chr  "00:00:00" "00:01:00" "00:02:00" "00:03:00" ...
#  $ Sub_metering_1: num  0 0 0 0 0 0 0 0 0 0 ...
#  $ Sub_metering_2: num  0 0 0 0 0 0 0 0 0 0 ...
#  $ Sub_metering_3: num  0 0 0 0 0 0 0 0 0 0 ...

##############
# Add Features
##############

df$year <- year(df$DateTime)
df$quarter <- quarter(df$DateTime)
df$month <- month(df$DateTime)
df$week <- week(df$DateTime)
df$day <- day(df$DateTime) #day of the month
df$wday <- wday(df$DateTime, label=FALSE)
#documentation is incorrect.  weekstart says Monday=1, but in reality Sunday=1 per their examples.
# supported in our data as well.  

#it may be useful to see the day of the week name, when we think 
#  about weekday/weekend in the customer electricity usage
df$wdayName <- wday(df$DateTime, label=TRUE, abbr=TRUE)

df$hour <- hour(df$DateTime)
df$minute <- minute(df$DateTime)

head(df)
summary(df)
#            Sub_metering_1   Sub_metering_2   Sub_metering_3     
#Min.   : 0.000   Min.   : 0.000   Min.   : 0.000   
#1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 0.000    
#Median : 0.000   Median : 0.000   Median : 1.000     
#Mean   : 1.159   Mean   : 1.343   Mean   : 6.216    
#3rd Qu.: 0.000   3rd Qu.: 1.000   3rd Qu.:17.000   
#Max.   :82.000   Max.   :78.000   Max.   :31.000

sd(df$Sub_metering_1) # [1] 6.288272
sd(df$Sub_metering_2) # [1] 5.972199
sd(df$Sub_metering_3) # [1] 8.341281
#individual hist don't show much, as the minute usage is low.  Will help more once we group by week, etc
hist(df$Sub_metering_1, breaks=25)
hist(df$Sub_metering_2, breaks=25)
hist(df$Sub_metering_3, breaks=25)

vioplot(df$Sub_metering_1, df$Sub_metering_2, df$Sub_metering_3, 
        names=c("SubMeter #1", "SubMeter #2", "SubMeter #3"), 
        ylab="energy (watt-hour)",
        col="lightblue")

any(is.na(df)) # True - there are 180 in DateTime
summary(df)
# Let's look at the NA values in DateTime (180)
head(filter(df, is.na(DateTime)))
#NA values look like a bad parsing job on 3/25/2007 and 3/29/2007 data
# DateTime       Date         Time 
# <NA>           2007-03-25   02:00:00 

#Should we remove these?

#################
# TASK 2
#################


#####################
# Filtering pipeline
#####################

##----Process Steps from Sub-setting to Graphing------#
# 1. dplyr::mutate(): add column with desired time interval (e.g., Yr/Mo/Day) using lubridate 
# 2. dplyr::filter(): select cols to filter by; full ds + added col from mutate are avail to filter at this stage
# 3. dplyr::group_by(): select which time intervals to subset and their order 
# 4. dplyr::summarize(): select the vars and any calculations for these vars
# 5. dplyr::first(): add DateTime col to end summary() string using first() (not needed for forecasting)
# 6. dplyr::filter() to remove any NA or narrow data ranges 

#############
## Subsets
#############

## Plot all of sub-meter 1 - this will take awhile
#plot(df$Sub_metering_1)

## Subset the second week of 2008 - All Observations
houseWeek <- filter(df, year == 2008 & week == 2)
## Plot subset houseWeek
plot(houseWeek$Sub_metering_1)

## Subset the 9th day of January 2008 - All observations
houseDay <- filter(df, year == 2008 & month == 1 & day == 9)

## Plot sub-meter 1
plot_ly(houseDay, x = ~houseDay$DateTime, y = ~houseDay$Sub_metering_1, type = 'scatter', mode = 'lines')

## Plot sub-meter 1, 2 and 3 with title, legend and labels - All observations 
plot_ly(houseDay, x = ~houseDay$DateTime, y = ~houseDay$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseDay$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseDay$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

## Subset the 9th day of January 2008 - 10 Minute frequency
houseDay10 <- filter(df, year == 2008 & month == 1 & day == 9 & 
                       (minute == 0 | minute == 10 | minute == 20 | minute == 30 | minute == 40 | minute == 50))
## Plot sub-meter 1, 2 and 3 with title, legend and labels - 10 Minute frequency
plot_ly(houseDay10, x = ~houseDay10$DateTime, y = ~houseDay10$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseDay10$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseDay10$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

######### Visualization of a single week
houseWeek <- filter(df, year == 2007 & week == 12)
plot_ly(houseWeek, x = ~houseWeek$DateTime, y = ~houseWeek$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseWeek$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseWeek$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption 12th week of 2007",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
houseWeekSample <- filter(df, year == 2007 & week == 12 & minute == 22 )
plot_ly(houseWeekSample, x = ~houseWeekSample$DateTime, y = ~houseWeekSample$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseWeekSample$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseWeekSample$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption 12th week of 2007",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

########## Custom visualization
houseWday <- filter(df, year == 2008 & wday==2)
plot_ly(houseWday, x = ~houseWday$DateTime, y = ~houseWday$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseWday$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseWday$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption Mondays in 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
houseWdaySample <- filter(df, year == 2008 & wday==2 & minute==10)
plot_ly(houseWdaySample, x = ~houseWdaySample$DateTime, y = ~houseWdaySample$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~houseWdaySample$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~houseWdaySample$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption Mondays in 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

## Optional Work: Pie chart showing total power use over a day by each sub-meter
#will need to stack the data first
houseDayStacked <- gather(houseDay, key='SubMeter', value='usage', Sub_metering_1, Sub_metering_2, Sub_metering_3)
plot_ly(houseDayStacked, labels = ~SubMeter, values = ~usage, type = 'pie') %>% 
  layout(title = "Power Consumption January 9th, 2008")

#total power usage over a year - unsampled
house2008 <- filter(df, year==2008)
house2008Stacked <- gather(house2008, key='SubMeter', value='usage', Sub_metering_1, Sub_metering_2, Sub_metering_3)
plot_ly(house2008Stacked, labels = ~SubMeter, values = ~usage, type = 'pie') %>% 
  layout(title = "Power Consumption in all of 2008")  
#was super quick... less than 5 sec to compute. 

dfStacked <- gather(df, key='SubMeter', value='usage', Sub_metering_1, Sub_metering_2, Sub_metering_3) 
plot_ly(dfStacked, labels = ~SubMeter, values = ~usage, type = 'pie') %>% 
  layout(title = "Power Consumption for 2007-2009") 

####################################
### Prepare to Analyze the Data ####
####################################

## Note: in conversion to TS, frequency is important.  This is how many data points per year you'll have.
## This is critical for extracting seasonal behavior


## Subset to one observation per week on Mondays at 8:00pm for 2007, 2008 and 2009
house070809weekly <- filter(df, wday == 2 & hour == 20 & minute == 1)
## Create TS object with SubMeter3 (freq=52 since 1 data point per week, 52 weeks in year)
tsSM3_070809weekly <- ts(house070809weekly$Sub_metering_3, frequency=52, start=c(2007,1))

autoplot(tsSM3_070809weekly)+
  labs(x="Time", y="Submeter_3 Power Usage", title="Weekly plot for Mondays @ 8pm")
plot.ts(tsSM3_070809weekly)
## Plot sub-meter 3
plot_ly(house070809weekly, x = ~house070809weekly$DateTime, y = ~house070809weekly$Sub_metering_3, type = 'scatter', mode = 'lines') %>%
  layout(title = "Mondays at 8:00pm for 2007, 2008 and 2009",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Sub-meter #3 Power (watt-hours)"))

#additional visualizations:
#tried monthly (day==1), but plot was too sparse
# sub-meter 1 for Saturdays at noon for 2007/8/9
house070809Sat <- filter(df, wday == 7 & hour == 12 & minute == 1)
ts_house070809Sat <- ts(house070809Sat$Sub_metering_1, frequency=52, start=c(2007,1))
plot.ts(ts_house070809Sat)
autoplot(ts_house070809Sat)+
  labs(x="Time", y="Submeter_1 Power Usage", title="Saturdays @ 12pm")
#this is pretty ugly.. sampling may be the problem here.

#daily plot for submeter_2 for 2008 at 6pm
house08daily <- filter(df, year == 2008 & hour == 18 & minute == 1)
ts_house08daily <-  ts(house08daily$Sub_metering_2, frequency=365, start=c(2008,1))
autoplot(ts_house08daily)+
  labs(x="Time", y="Submeter_2 Power Usage", title="Daily plot for 2008 @ 6pm")
plot.ts(ts_house08daily)
# this is also pretty ugly.. sampling on a single minute for consumption may be flawed


#let's do one that aggregates the data instead
#Daily Averaging
dfDaily <-  df %>%
  group_by(year, month, day) %>%
  summarize(meanSM1 = mean(Sub_metering_1),
            meanSM2 = mean(Sub_metering_2),
            meanSM3 = mean(Sub_metering_3),
            DateTime = first(DateTime))

dfDaily
plot_ly(dfDaily, x = ~dfDaily$DateTime, y = ~dfDaily$meanSM1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~dfDaily$meanSM2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~dfDaily$meanSM3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Daily averaged power consumption",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

# zoom in on july-oct 2008
plot_ly(dfDaily, x = ~dfDaily$DateTime, y = ~dfDaily$meanSM1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~dfDaily$meanSM2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~dfDaily$meanSM3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Daily averaged power consumption: 6/1/2008 - 10/1/2008",
         xaxis = list(title = "Time", range=c("2008-06-01", "2008-10-01")),
         yaxis = list (title = "Power (watt-hours)"))

#pretty noisy.. let's reduce to weekly averaging

#Weekly Averaging  ###THIS IS A KEY PLOT
dfWeekly <-  df %>%
  group_by(year, week) %>%
  summarize(meanSM1 = mean(Sub_metering_1),
            meanSM2 = mean(Sub_metering_2),
            meanSM3 = mean(Sub_metering_3),
            DateTime = first(DateTime))
dfWeekly
plot_ly(dfWeekly, x = ~dfWeekly$DateTime, y = ~dfWeekly$meanSM1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~dfWeekly$meanSM2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~dfWeekly$meanSM3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Weekly averaged power consumption",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

#### SM2 trend #####
ts_dfWeeklySM2 <- ts(dfWeekly$meanSM2, frequency=52, start=c(2007,1))
autoplot(ts_dfWeeklySM2)+
  labs(x="Time", y="Submeter_2 Power Usage", title="Weekly averaged usage")

### SM1 trend #####
ts_dfWeeklySM1 <- ts(dfWeekly$meanSM1, frequency=52, start=c(2007,1))
autoplot(ts_dfWeeklySM1)+
  labs(x="Time", y="Submeter_1 Power Usage", title="Weekly averaged usage")

### NOT USED - SM3 trend #########
ts_dfWeeklySM3 <- ts(dfWeekly$meanSM3, frequency=52, start=c(2007,1))
autoplot(ts_dfWeeklySM3)+
  labs(x="Time", y="Submeter_3 Power Usage", title="Weekly averaged usage")

#see the hourly trends averaged for all days in data set
df1dayagg <-  df %>%
  group_by(hour) %>%
  summarize(meanSM1 = mean(Sub_metering_1),
            meanSM2 = mean(Sub_metering_2),
            meanSM3 = mean(Sub_metering_3),
            DateTime = first(DateTime))
plot_ly(df1dayagg, x = ~df1dayagg$DateTime, y = ~df1dayagg$meanSM1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~df1dayagg$meanSM2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~df1dayagg$meanSM3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Daily power consumption profile",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))

#see the weekday trends averaged for all days in data set
df1weekagg <-  df %>%
  group_by(wday, hour) %>%
  summarize(meanSM1 = mean(Sub_metering_1),
            meanSM2 = mean(Sub_metering_2),
            meanSM3 = mean(Sub_metering_3),
            DateTime = first(DateTime),
            wdayName = first(wdayName))
plot_ly(df1weekagg, x = ~df1weekagg$DateTime, y = ~df1weekagg$meanSM1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~df1weekagg$meanSM2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~df1weekagg$meanSM3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Weekly power consumption profile",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
head(df1weekagg)

#%%%%%%%%%%%%%%%%%%%%%%%#
####### Forecasting #####
#%%%%%%%%%%%%%%%%%%%%%%%#

fitSM3 <- tslm(tsSM3_070809weekly ~ trend + season)
summary(fitSM3)
#Residual standard error: 9.046 on 104 degrees of freedom
#Multiple R-squared:  0.263,	Adjusted R-squared:  -0.1055 
#F-statistic: 0.7138 on 52 and 104 DF,  p-value: 0.9105

## Create the forecast for sub-meter 3. Forecast ahead 20 time periods
forecastfitSM3 <- forecast(fitSM3, h=20)
## Plot the forecast for sub-meter 3. 
plot(forecastfitSM3)
# default is 80% and 95% prediction intervals
## Create sub-meter 3 forecast with confidence levels 80 and 90
forecastfitSM3c <- forecast(fitSM3, h=20, level=c(80,90))
## Plot sub-meter 3 forecast, limit y and add labels
plot(forecastfitSM3c, ylim = c(0, 20), ylab= "SM3 Watt-Hours", xlab="Time")
forecastfitSM3c

#SM1: ts_house070809Sat
fitSM1 <- tslm(ts_dfWeeklySM1 ~ trend + season)
summary(fitSM1)
#Residual standard error: 0.4753 on 107 degrees of freedom
#Multiple R-squared:  0.4411,	Adjusted R-squared:  0.1694 
#F-statistic: 1.624 on 52 and 107 DF,  p-value: 0.018

## Create the forecast for sub-meter 2. Forecast ahead 20 time periods
forecastfitSM1 <- forecast(fitSM1, h=20)
## Plot the forecast for sub-meter 2. 
plot(forecastfitSM1, ylim = c(0, 2.5), ylab= "SM1 Watt-Hours", xlab="Time")

#SM2: ts_dfWeeklySM2
fitSM2 <- tslm(ts_dfWeeklySM2 ~ trend + season)
fitSM2
summary(fitSM2)
#Residual standard error: 0.5085 on 107 degrees of freedom
#Multiple R-squared:  0.4851,	Adjusted R-squared:  0.2349 
#F-statistic: 1.939 on 52 and 107 DF,  p-value: 0.002039

## Create the forecast for sub-meter 2. Forecast ahead 20 time periods
forecastfitSM2 <- forecast(fitSM2, h=20)
## Plot the forecast for sub-meter 2. 
plot(forecastfitSM2, ylim = c(0, 3), ylab= "SM2 Watt-Hours", xlab="Time")


#%%%%%%%%%%%%%%%%%%%%%%%#
### Decomposition #######
#%%%%%%%%%%%%%%%%%%%%%%%#

## Decompose Sub-meter 3 into trend, seasonal and remainder
components070809SM3weekly <- decompose(tsSM3_070809weekly)
## Plot decomposed sub-meter 3 
plot(components070809SM3weekly)
## Check summary statistics for decomposed sub-meter 3 
summary(components070809SM3weekly)
# comments: clear decrease in usage during 2nd half of year 

#Submeter 1
componentsWeeklySM1 <- decompose(ts_dfWeeklySM1)
plot(componentsWeeklySM1)
## Check summary statistics for decomposed sub-meter 1 
summary(componentsWeeklySM1)
# comments: mid summer shows a consistent dip

#submeter 2
componentsWeeklySM2 <- decompose(ts_dfWeeklySM2)
plot(componentsWeeklySM2)
## Check summary statistics for decomposed sub-meter 2 
summary(componentsWeeklySM2)
# comments: clear downward trend

## not used - weekly submeter 3
componentsWeeklySM3 <- decompose(ts_dfWeeklySM3)
plot(componentsWeeklySM3)


#summary stats
summary(components070809SM3weekly$x)
summary(components070809SM3weekly$seasonal)
summary(components070809SM3weekly$trend)
summary(components070809SM3weekly$random)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
### Holt-Winters Forecasting #######
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#Step 1: 
## Seasonal adjusting sub-meter 3 by subtracting the seasonal component & plot
tsSM3_070809Adjusted <- tsSM3_070809weekly - components070809SM3weekly$seasonal
autoplot(tsSM3_070809Adjusted)
## Test Seasonal Adjustment by running Decompose again. Note the very, very small scale for Seasonal
plot(decompose(tsSM3_070809Adjusted))
#notice that the values for seasonal are now very small (1E-15)
#reference:
plot(decompose(tsSM3_070809weekly))
## Holt Winters Exponential Smoothing & Plot
tsSM3_HW070809 <- HoltWinters(tsSM3_070809Adjusted, beta=FALSE, gamma=FALSE)
plot(tsSM3_HW070809, ylim = c(0, 25))

#alternatively, Holt-winters can be used with seasaonality included (but no trend so BETA=FALSE)
tsSM3_HW070809v2 <- HoltWinters(tsSM3_070809weekly, beta=FALSE)
plot(tsSM3_HW070809v2, ylim = c(0, 25))
#this appears very overfitted and not as good.  Apparently Holt-winters estimating the 
#seasonal component at a current time point is not as good

##### Forecasting ################
## HoltWinters forecast & plot
tsSM3_HW070809for <- forecast(tsSM3_HW070809, h=25)
plot(tsSM3_HW070809for, ylim = c(0, 20), ylab= "Watt-Hours", xlab="Time - Sub-meter 3")

## Forecast HoltWinters with diminished confidence levels (10% and 25%)
tsSM3_HW070809forC <- forecast(tsSM3_HW070809, h=25, level=c(50,80))
plot(tsSM3_HW070809forC, ylim = c(0, 23), ylab= "Watt-Hours", xlab="Time - Sub-meter 3")
## Plot only the forecasted area
plot(tsSM3_HW070809forC, ylim = c(0, 23), ylab= "Watt-Hours", xlab="Time - Sub-meter 3", start(2010))

### Repeat for SM1 and SM2
#Submeter 1
ts_dfWeeklySM1Adjusted <- ts_dfWeeklySM1 - componentsWeeklySM1$seasonal
autoplot(ts_dfWeeklySM1Adjusted)
plot(decompose(ts_dfWeeklySM1Adjusted))
#notice that the values for seasonal are now very small (1E-15)

## Holt Winters Exponential Smoothing & Plot
ts_dfHWSM1 <- HoltWinters(ts_dfWeeklySM1Adjusted, beta=FALSE, gamma=FALSE)
plot(ts_dfHWSM1, ylim = c(0, 3))
#again, doing seasonailty within HW isn't great - overfits. Perhaps because seasonality is overly complex
ts_dfHWSM1v2 <- HoltWinters(ts_dfWeeklySM1, beta=FALSE)
plot(ts_dfHWSM1v2, ylim = c(0, 3))

ts_dfHWSM1for <- forecast(ts_dfHWSM1, h=25)
plot(ts_dfHWSM1for, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 1")
ts_dfHWSM1forC <- forecast(ts_dfHWSM1, h=25, level=c(50,80))
plot(ts_dfHWSM1forC, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 1")
plot(ts_dfHWSM1forC, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 1", start(2010))

#Submeter 2
ts_dfWeeklySM2Adjusted <- ts_dfWeeklySM2 - componentsWeeklySM2$seasonal
autoplot(ts_dfWeeklySM2Adjusted)
plot(decompose(ts_dfWeeklySM2Adjusted))
#notice that the values for seasonal are now very small (1E-15)

## Holt Winters Exponential Smoothing & Plot
ts_dfHWSM2 <- HoltWinters(ts_dfWeeklySM2Adjusted, beta=FALSE, gamma=FALSE)
ts_dfHWSM2
plot(ts_dfHWSM2, ylim = c(0, 3))
#again, doing seasonailty within HW isn't great - overfits. Perhaps because seasonality is overly complex
ts_dfHWSM2v2 <- HoltWinters(ts_dfWeeklySM2)
plot(ts_dfHWSM2v2, ylim = c(0, 3)) #overfitted again

ts_dfHWSM2for <- forecast(ts_dfHWSM2, h=25)
plot(ts_dfHWSM2for, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 2")
ts_dfHWSM2forC <- forecast(ts_dfHWSM2, h=25, level=c(50,80))
plot(ts_dfHWSM2forC, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 2")
plot(ts_dfHWSM2forC, ylim = c(0, 3), ylab= "Watt-Hours", xlab="Time - Sub-meter 2", start(2010))
# it fails to capture the clear decreasing trend seen in the data, but if
# we remove beta=FALSE, it overfits and the forecasts get unreaslistically wide

### Alternative SM3 (weekly averaged) - should be an improvement

ts_dfWeeklySM3Adjusted <- ts_dfWeeklySM3 - componentsWeeklySM3$seasonal
ts_dfHWSM3 <- HoltWinters(ts_dfWeeklySM3Adjusted, beta=FALSE, gamma=FALSE)
ts_dfHWSM3forC <- forecast(ts_dfHWSM3, h=25, level=c(50,80))
plot(ts_dfHWSM3forC, ylim = c(0, 12), ylab= "Watt-Hours", xlab="Time - Sub-meter 3")
plot(ts_dfHWSM3forC, ylim = c(0, 12), ylab= "Watt-Hours", xlab="Time - Sub-meter 3", start(2010))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
### HW Forcast Evaluation    #######
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

acf(tsSM3_HW070809forC$residuals, lag.max=20, na.action=na.pass)
#low residuals for lag
Box.test(tsSM3_HW070809forC$residuals, lag=20, type="Ljung-Box")
# X-squared = 16.493, df = 20, p-value = 0.6856
# this p-value is good - > 0.05 (95% chance values are independent)
acf(ts_dfHWSM1forC$residuals, lag.max=20, na.action=na.pass)
#low residuals for lag
Box.test(ts_dfHWSM1forC$residuals, lag=20, type="Ljung-Box")
# X-squared = 12.799, df = 20, p-value = 0.8858
# this p-value is good - > 0.05 (95% chance values are independent)
acf(ts_dfHWSM2forC$residuals, lag.max=20, na.action=na.pass)
# several push the limits, barely
Box.test(ts_dfHWSM2forC$residuals, lag=20, type="Ljung-Box")
# X-squared = 37.761, df = 20, p-value = 0.009469
# this p-value is BAD (< 0.05) 
hist(ts_dfHWSM2forC$residuals)
#however, the residuals look normally distributed, so that's good.