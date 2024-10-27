library(ggplot2)
library(knitr)
library(formattable)

# Load files with regrets for different setting combinations

stoch_weak_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_weak_bandit_regrets.csv")
stoch_global_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_global_bandit_regrets.csv")
stoch_weak_full_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_weak_full_regrets.csv")

adv_weak_full_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_weak_full_regrets.csv")
stoch_adv_weak_full_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_weak_full_regrets.csv")
adv_weak_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_weak_bandit_regrets.csv")
stoch_adv_weak_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_weak_bandit_regrets.csv")

adv_global_full_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_global_full_regrets.csv")
stoch_adv_global_full_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_global_full_regrets.csv")
adv_global_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_global_bandit_regrets.csv")
stoch_adv_global_bandit_regrets <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_global_bandit_regrets.csv")

# Load files with profits for different setting combinations

stoch_weak_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_weak_bandit_profits.csv")
stoch_global_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_global_bandit_profits.csv")
stoch_weak_full_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_weak_full_profits.csv")

adv_weak_full_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_weak_full_profits.csv")
stoch_adv_weak_full_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_weak_full_profits.csv")
adv_weak_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_weak_bandit_profits.csv")
stoch_adv_weak_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_weak_bandit_profits.csv")

adv_global_full_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_global_full_profits.csv")
stoch_adv_global_full_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_global_full_profits.csv")
adv_global_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/adv_global_bandit_profits.csv")
stoch_adv_global_bandit_profits <- read.csv("/Users/basiakarwowska/Documents/BOCCONI/Thesis/Code/Codes updated/Final codes/Data/stoch_adv_global_bandit_profits.csv")

# Load files with profits for different setting combinations

# Create independent variables: time steps and transformed time steps
t <- 1:10000
t_23 <- t^(2/3)
t_sqrt <- sqrt(t)
t_log <- log(t)

# Create arrays with plot titles and data sets
plot_titles <- c("Stochastic Setting, Weak Budget Balance, Bandit Feedback", 
                 "Stochastic Setting, Global Budget Balance, Bandit Feedback", 
                 "Stochastic Setting, Weak Budget Balance, Full Feedback",
                 "Adversarial Setting, Weak Budget Balance, Full Feedback", 
                 "Semi-Adversarial Setting, Weak Budget Balance, Full Feedback",
                 "Adversarial Setting, Weak Budget Balance, Bandit Feedback",
                 "Semi-Adversarial Setting, Weak Budget Balance, Bandit Feedback",
                 "Adversarial Setting, Global Budget Balance, Full Feedback", 
                 "Semi-Adversarial Setting, Global Budget Balance, Full Feedback",
                 "Adversarial Setting, Global Budget Balance, Bandit Feedback",
                 "Semi-Adversarial Setting, Global Budget Balance, Bandit Feedback")
                 
data_sets_regrets <- list(stoch_weak_bandit_regrets, stoch_global_bandit_regrets, stoch_weak_full_regrets, adv_weak_full_regrets, 
               stoch_adv_weak_full_regrets, adv_weak_bandit_regrets, stoch_adv_weak_bandit_regrets, adv_global_full_regrets, 
               stoch_adv_global_full_regrets, adv_global_bandit_regrets, stoch_adv_global_bandit_regrets)

data_sets_profits <- list(stoch_weak_bandit_profits, stoch_global_bandit_profits, stoch_weak_full_profits, adv_weak_full_profits, 
                          stoch_adv_weak_full_profits, adv_weak_bandit_profits, stoch_adv_weak_bandit_profits, adv_global_full_profits, 
                          stoch_adv_global_full_profits, adv_global_bandit_profits, stoch_adv_global_bandit_profits)


for (i in 1:length(data_sets_regrets)) {
  
  # Data for the current setting
  regrets <- data_sets_regrets[[i]]$regret 
  acc_profits <- data_sets_profits[[i]]$profit
  plot_title <- plot_titles[i]
  
  # Linear fit
  linear_model <- lm(regrets ~ t)
  regret_linear_fit <- predict(linear_model)
  r2_linear <- summary(linear_model)$r.squared
  
  # Sublinear t^(2/3) fit
  sublinear_23_model <- lm(regrets ~ t_23)
  regret_sublinear_23_fit <- predict(sublinear_23_model)
  r2_sublinear_23 <- summary(sublinear_23_model)$r.squared
  
  # Sublinear sqrt(t) fit
  sublinear_12_model <- lm(regrets ~ t_sqrt)
  regret_sublinear_12_fit <- predict(sublinear_12_model)
  r2_sublinear_12 <- summary(sublinear_12_model)$r.squared
  
  # Logarithmic fit
  sublinear_log_model <- lm(regrets ~ t_log)
  regret_sublinear_log_fit <- predict(sublinear_log_model)
  r2_sublinear_log <- summary(sublinear_log_model)$r.squared
  
  # Set up a 1x2 plotting grid
  par(mfrow = c(1, 2), mar = c(5, 5, 2, 1), oma = c(0, 0, 4, 0)) 
  
  # Plot 1: Regret over time with linear and sublinear fits
  plot(t, regrets, type = "l", col = "blue", lwd = 2,
       xlab = "Round", ylab = "Regret", main = "Regret Over Time with Linear and Sublinear Fits")
  lines(t, regret_linear_fit, col = "red", lty = 2, lwd = 2)  # Linear fit
  lines(t, regret_sublinear_23_fit, col = "green", lty = 2, lwd = 2)  # t^(2/3) fit
  lines(t, regret_sublinear_12_fit, col = "yellow", lty = 2, lwd = 2)  # t^(1/2) fit
  lines(t, regret_sublinear_log_fit, col = "orange", lty = 2, lwd = 2)  # log(t) fit
  
  # Add legend
  legend("bottomright", legend = c("Regret", "Linear Fit", "Sublinear Fit t^(2/3)", "Sublinear Fit t^(1/2)", "Sublinear Fit log(t)"),
         col = c("blue", "red", "green", "yellow", "orange"), lty = 1:2, lwd = 2, cex = 0.8)
  
  # Plot 2: Accumulated profit over time
  plot(t, acc_profits, type = "l", col = "blue", lwd = 2,
       xlab = "Round", ylab = "Accumulated Profit", main = "Accumulated Profit Over Time")
  
  # Common title
  mtext(plot_title, outer = TRUE, cex = 1.5)
  
  # Reset plotting layout
  par(mfrow = c(1, 1))
  
  # Table of R^2
  r2_values <- data.frame(
    Model = c("Linear Fit", "Sublinear Fit t^(2/3)", "Sublinear Fit sqrt(t)", "Sublinear Fit log(t)"),
    R_squared = c(r2_linear, r2_sublinear_23, r2_sublinear_12, r2_sublinear_log)
  )
  print(kable(r2_values, format = "latex", booktabs = TRUE, col.names = c("Model", "R-squared"), align = 'c'))
  # kable(r2_values, format = "latex", col.names = c("Model", "R-squared"), align = 'c')
  # print(formattable(r2_values))
  # print(r2_values)
  
  # To find and print the summary of the model with the highest R^2
  model_list <- list(
    "Linear Fit" = list("model" = linear_model, "r_squared" = r2_linear),
    "Sublinear Fit t^(2/3)" = list("model" = sublinear_23_model, "r_squared" = r2_sublinear_23),
    "Sublinear Fit sqrt(t)" = list("model" = sublinear_12_model, "r_squared" = r2_sublinear_12),
    "Sublinear Fit log(t)" = list("model" = sublinear_log_model, "r_squared" = r2_sublinear_log)
  )
  
  # Step 3: Find the model with the highest R^2
  best_model_name <- names(which.max(sapply(model_list, function(x) x$r_squared)))
  
  # Step 4: Print the summary of the model with the highest R^2
  cat("Model with the highest R^2:", best_model_name, "\n\n")
  print(summary(model_list[[best_model_name]]$model))
}





