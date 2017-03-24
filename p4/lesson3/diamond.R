library(ggplot2)
library(gridExtra)

data(diamonds)

nrow(diamonds)


summary(diamonds)


p1 <- ggplot(data = diamonds, aes(x = price))
p2 <-
  p1 + geom_histogram(binwidth = 100) + scale_x_continuous(breaks = seq(0, 15000, 500))
p3 <-
  p1 + geom_histogram(binwidth = 5) + scale_x_continuous(breaks = seq(0, 15000, 50), limits = c(0, 2000))

grid.arrange(p2, p3, ncol = 1)
summary(diamonds$price)


nrow(subset(diamonds, price < 500))
nrow(subset(diamonds, price < 250))
nrow(subset(diamonds, price >= 15000))


ggplot(data = diamonds, aes(x = price)) + geom_histogram(binwidth = 20) +  facet_wrap( ~
                                                                                         cut)

by(diamonds$price, diamonds$cut, summary)
by(diamonds$price, diamonds$cut, max)
by(diamonds$price, diamonds$cut, min)
by(diamonds$price, diamonds$cut, median)

qplot(x = price, data = diamonds) + facet_wrap(~ cut, scales = "free")


diamonds$pricePerCarat = diamonds$price / diamonds$carat




ggplot(data = diamonds, aes(x = pricePerCarat)) + geom_histogram(binwidth = 1) + facet_wrap(~
                                                                                              cut, scales = 'free')



by(diamonds$price, diamonds$color, summary)

ggplot(data = diamonds, mapping = aes(x = color, y = price)) + geom_boxplot() + coord_cartesian(ylim = c(0, 7700))

IQR(subset(diamonds$price, diamonds$color == 'J'))
IQR(subset(diamonds$price, diamonds$color == 'D'))




by(diamonds$pricePerCarat, diamonds$color, summary)

ggplot(data = diamonds,
       mapping = aes(x = color, y = pricePerCarat)) + geom_boxplot() + coord_cartesian(ylim = c(0, 5500))



ggplot(mapping = aes(x = carat), data = diamonds) +
  geom_freqpoly(binwidth = .05) + scale_x_continuous(limits = c(0, 3.2), breaks = seq(0, 3.2, 0.1)) +
  geom_hline(linetype = 'F1', yintercept = 2000) +
  geom_vline(linetype = 'dashed',
             xintercept = c(0.1, 0.3, 0.8, 1.01, 1.6, 2.0, 3.0, 5.0))
