library(ggplot2)


CATEG = read.csv("input/subr_classification.csv")[,c('subreddit','category','subscribers_K')]
DF = read.csv("analysis/deletion_boundaries/S_grouped_deletions.csv")

DF = merge(DF, CATEG, by ="subreddit", all.x = TRUE)
DF
head(DF)

min_display = -1
max_display = min_display*-1
DF$threshold[DF$threshold< min_display] = min_display+0.1
DF$threshold[DF$threshold> max_display] = max_display-0.1
DF$min_threshold = ifelse(DF$effect  >0, DF$threshold, min_display)
DF$max_threshold = ifelse(DF$effect  <0, DF$threshold, max_display)
DF$category = factor(DF$category)

plot_boundary = function(DF){
  ggplot(DF) +
    aes(x = subreddit, y =quant_50) +
    geom_segment(aes(y = quant_05, yend = quant_95, x = subreddit, xend = subreddit), size = 8, col = c("dodgerblue"))+
    geom_segment(aes(y = min_threshold, yend = max_threshold, x = subreddit, xend = subreddit), size =1, col = c("black"))+
    geom_point(col = "white", size = 1) +
    labs(y = "Similarity with avg embedding", size =5)+
    theme_classic()+
    coord_flip() +
    ylim(c(min_display,max_display))
}

for (categ in levels(DF$category)){
  FILTERED = DF[DF$category==categ,]
  g = plot_boundary(FILTERED) + ggtitle(categ)
  print(g)
  ggsave(plot=g, filename = paste0("output/plots/",categ,".png"), 
         device = "png", units = "cm", width = 10, height = nrow(FILTERED)+2)
  }
