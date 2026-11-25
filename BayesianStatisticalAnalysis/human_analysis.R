# plotting for multimodal interaction colab
library(tidyverse)
library(tidybayes)
library(aida)
# use the aida-theme for plotting
theme_set(theme_aida())
# package for Bayesian data analysis
library(brms)
library(bayestestR)
# package for color universal design (optimized color palette)
library(oneclust)
library(psych) # for logistic function
library(paletteer)
library(xtable)  # for latex outputs

### general settings ------------------------------------------------------
# global color scheme
# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_fill_brewer(palette = "Pastel1")
}

scale_fill_discrete <- function(...) {
  scale_fill_brewer(palette = "Pastel1")
}

set.seed(161)

## getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

### Reading in files --------------------------------------------------------------
# preprocessed communication data
df_f2f <- read_csv("gesture_per_utt_df_f2f.csv")
df <- read_csv("gesture_per_utt_df.csv")

df$experiment <- rep("video", nrow(df))
df_f2f$experiment <- rep("f2f", nrow(df_f2f))
df_f2f$condition <- rep("f2f", nrow(df_f2f))

df_info <- read_csv("participant_info_to_share.csv") %>% 
  rename(pair_id = Pair,
         speaker = Speaker,
         condition = Condition)

df_video <- left_join(df, df_info[1:3], by=c("pair_id", "speaker"))

names(df_video)
names(df_f2f)
df_f2f <- df_f2f %>% dplyr::select(-accuracy)
df_all <- rbind(df_video, df_f2f)

### Preprocessing -----------------

# round variable: categorical
# converting into same format across experiments
df_all$round <- gsub("^R", "", df_all$round)
df_all$pair_id <- gsub("^pair", "", df_all$pair_id)

df_two_cond <- df_all %>% 
  filter(condition == "AO" | condition == "Sym")

df_two_cond$condition <- factor(df_two_cond$condition, 
                                levels = c("AO", "Sym"))

df_two_cond %>% 
  ggplot(
    aes(x=condition, y=gesture_present)) +
  geom_violin(aes(fill=condition)) + 
  stat_summary(fun = mean,
               geom = "point",
               shape = 23,
               size = 4,
               fill = 'grey') +
  labs(y = "Gesture presence",
       x = "Interaction condition")+
  theme(legend.position = "none")
ggsave("gesture_per_cond.jpg", width = 7.00, height = 4.67, units = "in")


# Priors ---------

priors <- c(
  set_prior("normal(-1, 0.5)", class = "Intercept"),
  set_prior("normal(0, 0.1)", class = "b"),
  set_prior("lkj(2)", class = "cor")
)


# Model 1: Gesture presence per condition ----------

fit_cond_hier <- brm(gesture_present ~ condition
                     + (condition | pair_id),
                     data = df_two_cond,
                     prior = priors,
                     family = bernoulli(link = logit),
                     cores = getOption("mc.cores", 4),
                     chains = 4,
                     iter = 4000,
                     warmup = 1000,
                     control = list(adapt_delta = 0.99),
                     seed = 201,
                     file = "fit_gesture_condition_two_cond_hierarchical")
summary(fit_cond_hier, priors = TRUE)
pairs(fit_cond_hier)
pp_check(fit_cond_hier)
pp_check(fit_cond_hier, "hist")
# more gestures in Sym than A0

posterior_summary <- describe_posterior(fit_cond_hier, centrality = "mean", ci_method = "hdi", 
                                        test = c("pd", "rope"), rope_range = c(-0.1, 0.1),
                                        diagnostic = c("ESS", "Rhat"))

posterior_summary$CrI <- paste0("[", round(posterior_summary$CI_low, 3), ", ", 
                                round(posterior_summary$CI_high, 3), "]")
posterior_summary <- posterior_summary[, !names(posterior_summary) %in% c("CI_low", "CI_high", "CI")]
posterior_summary$ROPE_Range <- paste0("[", round(posterior_summary$ROPE_low, 3), ", ", 
                                       round(posterior_summary$ROPE_high, 3), "]")
posterior_summary <- posterior_summary[, !names(posterior_summary) %in% c("ROPE_low", "ROPE_high", "ROPE_CI")]
posterior_summary$ROPE <- percent(posterior_summary$ROPE_Percentage)
posterior_summary$pd <- percent(posterior_summary$pd)
posterior_summary <- posterior_summary %>% 
  select(Parameter, Mean, CrI, pd, Rhat, ESS)

kable(posterior_summary, 
      format = "latex", 
      digits = 2,
      booktabs = TRUE, 
      caption = "Posterior Summary Statistics",
      align = "c",
      row.names = FALSE,
      label = "tab:gesture_present_condition")


# Model 2: Gesture presence per round -------

plot_round <- df_two_cond %>% 
  filter(!is.na(round)) %>% 
  ggplot(
    aes(x=round, y=gesture_present)) +
  geom_violin(aes(fill=round)) + 
  stat_summary(fun = mean,
               geom = "point",
               shape = 23,
               size = 4,
               fill = 'grey') +
  labs(y = "Gesture presence", 
       x = "Round") +
  theme(legend.position = "none")
plot_round
ggsave("round_gesture.jpg", width = 7.00, height = 4.67, units = "in")

# code round as numeric
df_two_cond$round <- as.numeric(df_two_cond$round)

fit_round_hier <- brm(gesture_present ~ round
                 + (round | pair_id),
                 data = df_two_cond,
                 prior = priors,
                 #sample_prior = "only",
                 family = bernoulli(link = logit),
                 cores = getOption("mc.cores", 4),
                 seed = 115,
                 iter = 4000,
                 warmup = 1000,
                 control = list(adapt_delta = 0.99),
                 file = "fit_gesture_round_num_hier")
summary(fit_round_hier)
pairs(fit_round_hier)
pp_check(fit_round_hier) # pretty good model fit
pp_check(fit_round_hier, type="hist") 
# fewer gestures in later rounds

posterior_summary <- describe_posterior(fit_round_hier, centrality = "mean", ci_method = "hdi", 
                                        test = c("pd", "rope"), rope_range = c(-0.1, 0.1),
                                        diagnostic = c("ESS", "Rhat"))

posterior_summary$CrI <- paste0("[", round(posterior_summary$CI_low, 3), ", ", 
                                round(posterior_summary$CI_high, 3), "]")
posterior_summary <- posterior_summary[, !names(posterior_summary) %in% c("CI_low", "CI_high", "CI")]
posterior_summary$ROPE_Range <- paste0("[", round(posterior_summary$ROPE_low, 3), ", ", 
                                       round(posterior_summary$ROPE_high, 3), "]")
posterior_summary <- posterior_summary[, !names(posterior_summary) %in% c("ROPE_low", "ROPE_high", "ROPE_CI")]
posterior_summary$ROPE <- percent(posterior_summary$ROPE_Percentage)
posterior_summary$pd <- percent(posterior_summary$pd)
posterior_summary <- posterior_summary %>% 
  select(Parameter, Mean, CrI, pd, Rhat, ESS)

kable(posterior_summary, 
      format = "latex", 
      digits = 2,
      booktabs = TRUE, 
      caption = "Posterior Summary Statistics",
      align = "c",
      row.names = FALSE,
      label = "gesture_present")
