setwd("/home/j0hndoe/Documents/git/reddit-disinformation/analysis")

library(tidyverse)
library(magrittr)

SUBR = read_csv("../input/subr_classification.csv") %>% 
        select(subreddit, category, subscribers = subscribers_K, postsperday = total)
PAIRS = read_csv("../output/n2v_use_el_month_2.csv") %>% 
          rename(culture = use_dist,
                 structure = euc_dist)

PAIRS %>% mutate_if(is_character, as_factor) %>% summary



#PAIRS %$% plot(euc_dist, use_dist)


FILT = PAIRS %>% filter(user_type=="home", removed_type == "kept") %>%
          left_join(SUBR %>% rename(category_i=category, subscribers_i=subscribers, postsperday_i=postsperday), 
                    by = c("i"="subreddit")) %>% 
          left_join(SUBR %>% rename(category_j=category, subscribers_j=subscribers, postsperday_j=postsperday), 
                    by = c("j"="subreddit")) %>% 
          mutate(subscribers_total = subscribers_i + subscribers_j,
                 subscribers_diff = abs(subscribers_i - subscribers_j),
                 postsperday_total = postsperday_i + postsperday_j,
                 postsperday_diff = abs(postsperday_i - postsperday_j))


FILT %$% cor.test(culture, structure)

FILT %>% select_if(is.numeric) %>% cor %>% round(3)

ggplot(FILT)+
  aes(x = structure,
      y = culture) +
  geom_point(col = "orange", alpha = 0.2, size = 1) +
  geom_smooth() +
  theme_classic()



ggplot(FILT)+
  aes(x = structure,
      y = culture,
      col = subscribers_total) +
  geom_point(size = 1) +
  geom_smooth() +
  theme_classic()


FILT$subscr5cat = cut(FILT$postsperday_total, quantile(FILT$postsperday_total), include.lowest = TRUE)
ggplot(FILT)+
  aes(x = structure,
      y = culture,
      col = subscribers_total) +
  geom_point(size = 1) +
  geom_smooth(method = "lm") +
  facet_wrap(~same_category)+
  theme_classic()




ggplot(FILT)+
  aes(x = culture,
      y = structure) +
  geom_point(col = "orange", alpha = 0.2, size = 1) +
  geom_smooth() +
  theme_classic()

for (categ in unique(c(FILT$category_i, FILT$category_j))){
  FILT[[paste0("CAT_",categ)]] = 0
  FILT[[paste0("CAT_",categ)]][FILT$category_i == categ | FILT$category_j == categ] = 1
  print(categ)
}

paste(FILT %>% select(starts_with("CAT_")) %>% names, collapse = " + ")

FILT$same_category = as.numeric(FILT$category_i == FILT$category_j)

write_rds(FILT, "cultsruct_filt.rds")





### Regression

REG = FILT %>% 
        select(i,j, month, structure, culture, subscribers_total , subscribers_diff, postsperday_total, postsperday_diff, same_category, contains("CAT_")) %>% 
        mutate_at(vars(culture, structure, subscribers_total , subscribers_diff, postsperday_total, postsperday_diff), scale)
summary(REG)

write_rds(REG, "cultsruct_reg.rds")
write_csv(REG, "cultsruct_reg.csv")

regC0 = lm(culture ~ structure, data = REG)
regC1 = lm(culture ~ structure + month + subscribers_total + subscribers_diff + postsperday_total + postsperday_diff + same_category , data = REG)
regC2 = lm(culture ~ structure + month + subscribers_total + subscribers_diff + postsperday_total + postsperday_diff + same_category +
                    CAT_business_economics + CAT_science_health + CAT_generic_news + CAT_generic_politics + CAT_local + CAT_technology + CAT_coronavirus, data = REG)


regC1i = lm(culture ~ structure* (month + subscribers_total + subscribers_diff + postsperday_total + postsperday_diff + same_category) , data = REG)
summary(regC1i)


regCm = lm(culture ~ structure * factor(month) + subscribers_total + subscribers_diff + postsperday_total + postsperday_diff + same_category , data = REG)
summary(regCm)

plot(regCm)

summary(regC0)
summary(regC1)
summary(regC2)

plot(regC1)
plot(regC2,1)

regCx = lm(culture ~ structure + factor(month) + subscribers_total, data = REG)
plot(regCx,1)

library(broom)

tidy(regC0) %>% mutate_at(vars(estimate:p.value), ~round(.,3))
tidy(regC1) %>% mutate_at(vars(estimate:p.value), ~round(.,3))


regS0 = lm(structure ~ culture, data = REG)
regS1 = lm(structure ~ culture + month + subscribers_total + subscribers_diff + postsperday_total + postsperday_diff + same_category +
                       CAT_business_economics + CAT_science_health + CAT_generic_news + CAT_generic_politics + CAT_local + CAT_technology + CAT_coronavirus, data = REG)
summary(regS0)
summary(regS1)

tidy(regS0) %>% mutate_at(vars(estimate:p.value), ~round(.,3))
tidy(regS1) %>% mutate_at(vars(estimate:p.value), ~round(.,3))
