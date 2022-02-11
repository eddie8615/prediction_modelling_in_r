# Introduction to R

# Import the crime.dta file ---------------------------------------------------

library(foreign)
library(tidyverse)
library(haven)

crime <- read.dta("crime.dta")

# Check the dimension of the data file
dim(crime)

# View the first few rows of data
head(crime)

# Summary
summary(crime)

# Scatterplot with linear fit -------------------------------------------------

# Base R

plot(crime ~ poverty, data = crime)
abline(lm(crime ~ poverty, data = crime), col = "red")

# ggplot2

library(ggplot2)
ggplot(crime, aes(x = poverty, y = crime)) +
    geom_point() +
    geom_smooth(method = "lm", color = "red")

## Histogram ------------------------------------------------------------------

ggplot(crime, aes(x = crime)) +
    geom_histogram(bins = 20) +
    facet_wrap(~ urban)

# We can color the bars according to their height:

ggplot(crime, aes(x = crime, fill= ..count..)) +
    geom_histogram(bins = 20) +
    facet_wrap(~ urban) +
  scale_fill_gradient("Count", low="green", high="red")

# Neither of the above plots are particularly helpful. Let's try plotting the
# densities instead:

ggplot(crime, aes(x = crime, group = urban, fill = urban)) +
    geom_density(alpha = 0.5) 

## Export (base graphics) -----------------------------------------------------

png("crime_histogram_base.png", height=600, width=700)
hist(crime$crime)
dev.off()

## Export (ggplot2) -----------------------------------------------------------

# 'Save' the plot as a new object

density_plot <- ggplot(crime, aes(x = crime, group = urban, fill = urban)) +
    geom_density(alpha = 0.5) 

# Save, using "ggsave"

ggsave(density_plot, device = "png", file = "crime_histogram.png")

# END.
