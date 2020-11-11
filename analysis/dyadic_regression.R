# install.packages("amen")
# https://cran.r-project.org/web/packages/amen/vignettes/amen.pdf 

library(tidyverse)
library(amen)

data(IR90s)

setwd("/home/j0hndoe/Documents/git/reddit-disinformation/analysis/")
REG = read_csv("cultsruct_reg.csv") 

REG04 = REG %>% filter(month==4) %>% arrange(i,j)
allsubr = sort(unique(c(REG04$i, REG04$j)))

#?pivot_wider
#Y = REG04 %>% select(i,j,culture) %>% pivot_wider(id_cols = i, names_from = j, values_from = culture)

create_dyad_matrix = function(DF, rownamez, varname) {
  Y = matrix(NA, length(rownamez), length(rownamez),
             dimnames = list(rownamez, rownamez))
  for (i in 1:length(rownamez))
    for (j in 1:length(rownamez)){
      icategory = rownamez[i]
      jcategory = rownamez[j]
      Y[i,j] = ifelse(length(DF[[varname]][DF$i==icategory & DF$j==jcategory])>0,
                      DF[[varname]][DF$i==icategory & DF$j==jcategory], 
                      ifelse(length(DF[[varname]][DF$j==icategory & REG04$i==jcategory])>0,
                             DF[[varname]][DF$j==icategory & DF$i==jcategory], NA))
      Y[j,i] = Y[i,j]
    }
  return(Y)
  }

create_dyad_covariates = function(DF, rownamez, covariates) {
  X = array(NA, 
            dim = c(length(rownamez), length(rownamez), length(covariates)),
            dimnames = list(rownamez, rownamez, covariates))
  for (c in 1:length(covariates))
    for (i in 1:length(rownamez))
      for (j in 1:length(rownamez)){
        icategory = rownamez[i]
        jcategory = rownamez[j]
        varname = covariates[c]
        X[i,j,c] = ifelse(length(DF[[varname]][DF$i==icategory & DF$j==jcategory])>0,
                        DF[[varname]][DF$i==icategory & DF$j==jcategory], 
                        ifelse(length(DF[[varname]][DF$j==icategory & REG04$i==jcategory])>0,
                               DF[[varname]][DF$j==icategory & DF$i==jcategory], NA))
        X[j,i,c] = X[i,j,c]
      }
    return(X)
  }

#REG04 = REG %>% filter(month==2) %>% arrange(i,j)

Y = create_dyad_matrix(REG04, allsubr, "culture")
Y[1:5,1:5]

Xd = create_dyad_covariates(REG04, allsubr, "structure")
Xd

Xdc1 = create_dyad_covariates(REG04, allsubr, c("structure", "subscribers_total", "subscribers_diff"))
Xdc2 = create_dyad_covariates(REG04, allsubr, c("structure", "subscribers_total", "subscribers_diff", 
                                                "postsperday_total", "postsperday_diff", "same_category"))
Xdc1

Xdc1[1:5,1:5,1]
Xdc1[1:5,1:5,2]
Xdc1[1:5,1:5,3]

### AME only with Y=culture
dyadic_01_cult = ame(Y)

summary(dyadic_01_cult)


### AME only with Y=culture and X=structure
dyadic_02_cult_struct = ame(Y=Y, Xd = Xd, symmetric = TRUE)
summary(dyadic_02_cult_struct)

  ### AME with Y=culture and Xd={structure , subscribers_total, subscribers_diff}
dyadic_03_cult_struct_ctrl = ame(Y=Y, Xd = Xdc1, symmetric = TRUE)
dyadic_03_cult_struct_ctrl2 = ame(Y=Y, Xd = Xdc2, symmetric = TRUE)

summary(dyadic_03_cult_struct_ctrl)
summary(dyadic_03_cult_struct_ctrl2)

