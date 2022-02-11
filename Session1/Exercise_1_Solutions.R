# Introduction to R


# Use the rnorm() function to generate a random sample of size 100 from a
# normal distribution with mean 10 and standard deviation 4
# -----------------------------------------------------------------------

vec <- rnorm(n = 100, 
             mean = 10, 
             sd = 5)


# Write a loop that adds numbers from 1 to 10 (i.e. 1 + 2 + 3 + ...)
# ------------------------------------------------------------------

result <- 0
for (i in 1:10) {
    result <- i + result
}
print(result)
