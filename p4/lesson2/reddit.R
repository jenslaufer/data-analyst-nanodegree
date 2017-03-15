reddit <- read.csv('reddit.csv')

table(reddit$employment.status)

str(reddit)

summary(reddit)
levels(reddit$age.range)

library(ggplot2)


qplot(data = reddit, x = ordered(age.range,c("Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or Above")))
qplot(data = reddit, x = income.range)
