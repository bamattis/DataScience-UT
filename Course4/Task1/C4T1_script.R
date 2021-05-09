
# DataScience Course 4, Task 1.

# Goal: 


# Secondary Goal:


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

## Lists attributes contained in a table
dbListFields(con,'iris')
#[1] "id" "SepalLengthCm" "SepalWidthCm"  "PetalLengthCm" "PetalWidthCm"  "Species" 

## Use asterisk to specify all attributes for download
irisALL <- dbGetQuery(con, "SELECT * FROM iris")

## Use attribute names to specify specific attributes for download
irisSELECT <- dbGetQuery(con, "SELECT SepalLengthCm, SepalWidthCm FROM iris")

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

any(is.na(df)) # True - there are 180.  
#Consider replacing with the nearest non-NA value to keep time domain continuous.
