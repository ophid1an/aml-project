library('scmamp')

df <- read.csv('erwthma1-results.csv')
df <- df[-c(1)]

imanDavenportTest(df)

nm <- nemenyiTest(df, alpha = 0.05)
nm

nm$diff.matrix

plotCD(results.matrix = df, alpha = 0.05)

fb <- postHocTest(data = df, test = 'friedman', correct = 'bergmann')
fb

writeTabular(table = fb$corrected.pval)

bold <- fb$corrected.pval < 0.05
bold[is.na(bold)] <- FALSE
writeTabular(table = fb$corrected.pval, format = 'f', bold = bold)