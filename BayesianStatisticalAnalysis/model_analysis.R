# plotting for multimodal interaction colab
library(tidyverse)
library(tidybayes)
library(aida)
# use the aida-theme for plotting
theme_set(theme_aida())
# package for Bayesian data analysis
library(brms)
library(bayestestR)
library(psych) # for logistic function
library(paletteer) # color pallette
library(knitr) # for latex outputs
library(scales)
### general settings------------------------------------------------------
# global color scheme / non-optimized
project_colors <- c("#f39100", "#8ecbd5", "#116966")

# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = project_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = project_colors)
}

set.seed(213)

## getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

### reading in files and preprocessing-----------------
df_all <- read.csv("corrected_merged_analysis_df.csv")

# 1) Analysis of communication setting--------------
df_select <- df_all %>% 
  filter(alignment_targets == "pose:visual" | alignment_targets == "none") %>% 
  select(Embedding.Type, Fold, seed, input_mode, pose_backbone, pose_layout, correct, 
         gold_label, predicted_label, Condition, round, top1_correct, top2_correct,
         top3_correct, top4_correct, top5_correct, alignment_loss_weight,
         alignment_targets) %>% 
  filter(Condition != "Asym") %>% 
  mutate(input_mode = factor(input_mode, levels = c("pose", "speech", "both")),
         pose_encoder = paste(pose_layout, pose_backbone, sep = "_"),
         Embedding.Type = factor(ifelse(Embedding.Type == "Gesture + Speech", "Fusion*",
                                        Embedding.Type), levels = c("Gesture", "Speech", "Fusion*"))) %>% 
  filter(pose_encoder == "unisign69_stgcn" | input_mode == "speech") %>% 
  # we use one speech model, so renaming it here
  mutate(pose_encoder = ifelse(input_mode == "speech", "unisign69_stgcn", pose_encoder)) %>% 
  filter(alignment_loss_weight != 0.1 & alignment_loss_weight != 0.2)

df_gesture <- df_select %>% 
  filter(input_mode == "pose",
         alignment_targets == "none")
df_fusion <- df_select %>% 
  filter(input_mode == "both",
         alignment_targets == "pose:visual")
df_speech <- df_select %>% 
  filter(input_mode == "speech")
df_select <- rbind(df_gesture, df_speech, df_fusion)

df_acc <- df_select %>% 
  group_by(Embedding.Type, Fold, seed, input_mode, pose_encoder, 
           Condition, alignment_loss_weight, alignment_targets) %>% 
  summarize(accuracy_mean = mean(gold_label == predicted_label),
            accuracy_sd = sd(gold_label == predicted_label))

df_acc_agr <- df_acc %>% 
  group_by(Embedding.Type, Condition, pose_encoder) %>% 
  summarize(mean_acc = mean(accuracy_mean),
            sd_acc = sd(accuracy_sd)) %>% 
  print()

df_acc_agr %>% 
  ggplot(aes(x = Condition, y = mean_acc, fill=Embedding.Type)) +
  geom_col(aes(y = mean_acc), position = "dodge2") +
  geom_errorbar(aes(ymin = mean_acc - sd_acc,
                    ymax = mean_acc + sd_acc), position = "dodge") +
  labs(y = "Accuracy",
       fill = "Input Mode")
ggsave("comm_cond_mmpose69_pv03.jpg", width = 7.00, height = 4.67, units = "in")

# plotting top-k accuracy
df_topk <- df_select %>% 
  group_by(Embedding.Type, 
           Condition) %>% 
  summarize(top_1 = mean(top1_correct),
            top_2 = mean(top2_correct),
            top_3 = mean(top3_correct),
            top_4 = mean(top4_correct),
            top_5 = mean(top5_correct)) %>% 
  pivot_longer(
    cols = starts_with("top_"),  # Select columns that start with "top_"
    names_to = "top_k",          # Name of the new column for the original column names
    values_to = "accuracy"          # Name of the new column for the values
  ) %>% 
  mutate(top_k = as.numeric(gsub("top_", "", top_k))) %>% 
  ungroup()

df_topk %>% 
  ggplot(aes(x = top_k, y = accuracy, color = Embedding.Type, linetype = Condition)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    x = "Top-k",
    y = "Accuracy",
    color = "Embedding Type",
    linetype = "Condition"
  ) +
  #scale_color_brewer(palette = "Set1") +  # Use a nice color palette
  scale_linetype_manual(values = c("AO" = "solid", "Sym" = "dashed")) +
  theme(
    legend.position = "right",
    #legend.title = element_text(size = 12, face = "bold"),
    #legend.text = element_text(size = 10)
  )
ggsave("topk_mmpose69_pv03.jpg", width = 7.00, height = 4.67, units = "in")

## binary logistic regression model across all pose encoders---------
# df for modeling
df_model <- df_select %>% 
  mutate(Condition = factor(Condition, levels = c("AO", "Sym")),
         input_mode = factor(input_mode, levels = c("pose", "speech", "both"))) 

fit_comm <- brm(correct ~ Condition * input_mode,
                data = df_model,
                family = bernoulli(link = "logit"), 
                cores = getOption("mc.cores", 4),
                file = "fit_comm_cond_mmpose69_pv03")
summary(fit_comm)

pp_check(fit_comm, "bars_grouped", group="Condition")
pp_check(fit_comm, "bars_grouped", group="input_mode")

### general results-----------
describe_posterior(fit_comm, rope_range = c(-0.01, 0.01))
posterior_summary <- describe_posterior(fit_comm, centrality = "mean", ci_method = "hdi", 
                                        test = c("pd", "rope"), rope_range = c(-0.01, 0.01),
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
  select(Parameter, Mean, CrI, pd, ROPE_Range, ROPE, Rhat, ESS)

kable(posterior_summary, 
      format = "latex", 
      digits = 2,
      booktabs = TRUE, 
      caption = "Posterior Summary Statistics Interaction condition",
      align = "c",
      row.names = FALSE,
      label = "posterior_summary_comm")

### plotting the estimated posterior mean accuracy------------
# extract posteriors
posteriors_fit_comm <- fit_comm %>%
  spread_draws(b_Intercept, 
               b_ConditionSym,
               b_input_modespeech,
               b_input_modeboth,
               `b_ConditionSym:input_modespeech`,
               `b_ConditionSym:input_modeboth`) %>%
  # calculate posteriors for each individual level
  mutate(ao_pose = b_Intercept,
         sym_pose = b_Intercept + b_ConditionSym,
         ao_speech = b_Intercept + b_input_modespeech,
         ao_both = b_Intercept + b_input_modeboth,
         sym_speech = b_Intercept + b_ConditionSym + b_input_modespeech + `b_ConditionSym:input_modespeech`,
         sym_both = b_Intercept + b_ConditionSym + b_input_modeboth + `b_ConditionSym:input_modeboth`) %>% 
  select(ao_pose, sym_pose, 
         ao_speech, sym_speech, 
         ao_both, sym_both) %>% 
  gather(key = "parameter", value = "posterior") %>% 
  mutate(Condition = ifelse(parameter == "ao_pose" | parameter == "ao_speech" | parameter == "ao_both", "AO", "Sym"),
         InputMode = ifelse(parameter == "ao_pose" | parameter == "sym_pose", "Pose",
                            ifelse(parameter == "ao_speech" | parameter == "sym_speech", "Speech", "Both")),
         posterior = logistic(posterior)) 

posteriors_agr_fit_comm <- posteriors_fit_comm %>% 
  group_by(Condition, InputMode) %>% 
  summarise(mean_posterior = mean(posterior),
            `95lowerCrI` = HDInterval::hdi(posterior, credMass = 0.95)[1],
            `95higherCrI` = HDInterval::hdi(posterior, credMass = 0.95)[2])

posteriors_agr_fit_comm %>% 
  mutate(InputMode = factor(InputMode, levels = c("Pose", "Speech", "Both"))) %>% 
  ggplot(aes(x = Condition, y = mean_posterior, fill=InputMode)) +
  geom_col(aes(y = mean_posterior), position = "dodge2") +
  geom_errorbar(aes(ymin = `95lowerCrI`, ymax = `95higherCrI`),
                #width = 0.3, 
                color = "black", position = "dodge2") +  
  labs(y = "Mean Posterior", fill = "Input Mode")
ggsave("posterior_comm_cond_mmpose69.jpg", width = 7.00, height = 4.67, units = "in")

### calculating specific differences----------
ao_pose <- posteriors_fit_comm %>% 
  filter(Condition == "AO",
         InputMode == "Pose") %>% 
  select(posterior) 
sym_pose <- posteriors_fit_comm %>% 
  filter(Condition == "Sym",
         InputMode == "Pose") %>% 
  select(posterior) 
describe_posterior(sym_pose - ao_pose, rope_range = c(-0.01, 0.01))
# substantial difference between ao-pose and sym-pose

ao_speech <- posteriors_fit_comm %>% 
  filter(Condition == "AO",
         InputMode == "Speech") %>% 
  select(posterior) 
sym_speech <- posteriors_fit_comm %>% 
  filter(Condition == "Sym",
         InputMode == "Speech") %>% 
  select(posterior) 
describe_posterior(sym_speech - ao_speech, rope_range = c(-0.01, 0.01))
# no substantial difference between ao-speech and sym-speech

ao_both <- posteriors_fit_comm %>% 
  filter(Condition == "AO",
         InputMode == "Both") %>% 
  select(posterior) 
sym_both <- posteriors_fit_comm %>% 
  filter(Condition == "Sym",
         InputMode == "Both") %>% 
  select(posterior) 
describe_posterior(sym_both - ao_both, rope_range = c(-0.01, 0.01))
# no substantial difference between ao-both and sym-both

# 2) Round-based analysis----------
df_acc <- df_select %>% 
  group_by(Embedding.Type, Fold, seed, input_mode, pose_encoder,
           round) %>% 
  summarize(accuracy_mean = mean(gold_label == predicted_label),
            accuracy_sd = sd(gold_label == predicted_label))

df_acc_agr <- df_acc %>% 
  group_by(Embedding.Type, round) %>% 
  summarize(mean_acc = mean(accuracy_mean),
            sd_acc = sd(accuracy_sd)) %>% 
  print()

df_acc_agr %>% 
  ggplot(aes(x = round, y = mean_acc, color=Embedding.Type)) +
  geom_point(aes(y = mean_acc)) +
  geom_errorbar(aes(ymin = mean_acc - sd_acc,
                    ymax = mean_acc + sd_acc), width=0.2) +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 6, by = 1)) +
  labs(y = "Accuracy",
       color = "Input Mode")
ggsave("round_mmpose69_pv03.jpg", width = 7.00, height = 4.67, units = "in")

## binary logistic regression model across all pose encoders---------
# df for modeling
df_model <- df_select %>% 
  mutate(input_mode = factor(input_mode, levels = c("pose", "speech", "both"))) 

fit_round <- brm(correct ~ input_mode * round,
                data = df_model,
                family = bernoulli(link = "logit"), 
                cores = getOption("mc.cores", 4),
                file = "fit_round_mmpose69_pv03")
summary(fit_round)

pp_check(fit_round, "bars_grouped", group="round")
pp_check(fit_round, "bars_grouped", group="input_mode")

### general results-----------
describe_posterior(fit_round, rope_range = c(-0.01, 0.01))
posterior_summary <- describe_posterior(fit_round, centrality = "mean", ci_method = "hdi", 
                                        test = c("pd", "rope"), rope_range = c(-0.01, 0.01),
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
  select(Parameter, Mean, CrI, pd, ROPE_Range, ROPE, Rhat, ESS)

kable(posterior_summary, 
      format = "latex", 
      digits = 2,
      booktabs = TRUE, 
      caption = "Posterior Summary Statistics Round",
      align = "c",
      row.names = FALSE,
      label = "tab:posterior_summary_round")