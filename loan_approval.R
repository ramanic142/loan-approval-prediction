# ===============================
# LOAN APPROVAL PREDICTION SCRIPT
# Proposal referenced for study goals and variables :contentReference[oaicite:1]{index=1}
# ===============================

# Install needed libraries once if not installed
# install.packages(c("readr","dplyr","ggplot2","reshape2","caret","car",
#                    "pROC","ResourceSelection","neuralnet","scales"))

# Load libraries
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(car)
library(pROC)
library(ResourceSelection)
library(neuralnet)
library(scales)

# ------------------------------
# STEP 1: IMPORT DATA
# ------------------------------
df <- read_csv("C:/Users/raman/Downloads/loan_approval_dataset.csv")

# View sample & structure
head(df)
str(df)  #data type of each column (numeric, text, factor, etc.)
summary(df) #Minimum, Maximum, Mean, Median for numeric variables

# ------------------------------
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ------------------------------
df <- df %>%
  mutate(    #mutate() creates new columns or changes existing ones.
    loan_status = tolower(trimws(loan_status)),
    loan_status = ifelse(loan_status %in% c("approved","1","yes"), 1, 0),
    loan_status = factor(loan_status, levels = c(0,1),
                         labels = c("Rejected","Approved")),
    
    education = as.factor(education),
    self_employed = as.factor(self_employed),
    
    total_assets = residential_assets_value +
      commercial_assets_value +
      luxury_assets_value +
      bank_asset_value,
    
    loan_to_income = loan_amount / (income_annum + 1)
  )

# Verify encoding
str(df)
summary(df$loan_status)

# ------------------------------
# STEP 3: TRAIN TEST SPLIT (75/25)
# ------------------------------
set.seed(123)
train_index <- createDataPartition(df$loan_status, p = 0.75, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# ------------------------------
# STEP 4: LOGISTIC REGRESSION MODEL
# ------------------------------
log_model <- glm(
  loan_status ~ income_annum + loan_amount + loan_term +
    cibil_score + no_of_dependents +
    education + self_employed +
    total_assets + loan_to_income,
  data = train,
  family = binomial
)

# Model summary
summary(log_model)

# Odds ratio + confidence interval
exp(cbind(Odds_Ratio = coef(log_model), confint.default(log_model)))

# VIF multicollinearity test
vif(log_model)

# ------------------------------
# STEP 5: LOGISTIC REGRESSION PREDICTIONS & METRICS
# ------------------------------
test$pred_prob  <- predict(log_model, newdata = test, type = "response")
test$pred_class <- ifelse(test$pred_prob > 0.5, "Approved", "Rejected")
test$pred_class <- factor(test$pred_class, levels = c("Rejected","Approved"))

# Confusion matrix
confusionMatrix(test$pred_class, test$loan_status, positive="Approved")

# ROC curve + AUC
roc_log <- roc(test$loan_status, test$pred_prob)
plot(roc_log, main="ROC Curve - Logistic Regression", print.auc=TRUE)
auc(roc_log)

# Hosmer-Lemeshow test
hl_y <- ifelse(train$loan_status == "Approved", 1, 0)
hoslem.test(hl_y, fitted(log_model), g=10)

# Cook's distance outlier influence
plot(cooks.distance(log_model), type="h", main="Cook's Distance - Logistic Model")
abline(h = 4/(nrow(train)-length(coef(log_model))), lty=2)

# ------------------------------
# STEP 6: NEURAL NETWORK MODEL
# ------------------------------
# Prepare numeric target 0/1 for NN
train$loan_status_num <- ifelse(train$loan_status == "Approved", 1, 0)
test$loan_status_num  <- ifelse(test$loan_status  == "Approved", 1, 0)

set.seed(123)
nn_model <- neuralnet(
  loan_status_num ~ income_annum + loan_amount + loan_term +
    cibil_score + no_of_dependents + total_assets,
  data = train,
  hidden = 3,
  linear.output = FALSE
)

# Plot NN architecture
plot(nn_model)

# Neural network predictions
nn_results <- compute(nn_model,
                      test[, c("income_annum","loan_amount","loan_term",
                               "cibil_score","no_of_dependents","total_assets")])

prob_nn <- as.vector(nn_results$net.result)

test$nn_pred <- ifelse(prob_nn > 0.5, "Approved","Rejected")
test$nn_pred <- factor(test$nn_pred, levels = c("Rejected","Approved"))

# Confusion matrix NN
confusionMatrix(test$nn_pred, test$loan_status, positive="Approved")

# ROC + AUC for NN
roc_nn <- roc(test$loan_status_num, prob_nn)
plot(roc_nn, main="ROC Curve - Neural Network", print.auc=TRUE)
auc(roc_nn)

# ------------------------------
# STEP 7: MODEL COMPARISON ROC
# ------------------------------
plot(roc_log, main="ROC Comparison: Logistic vs Neural Network")
lines(roc_nn, lty=2)
legend("bottomright", legend=c("Logistic Regression","Neural Network"), lty=c(1,2))

# ------------------------------
# STEP 8: KEY EDA PLOTS FOR PPT
# ------------------------------
# Income vs Loan Amount
ggplot(df, aes(x = income_annum, y = loan_amount, color = loan_status)) +
  geom_point(alpha = 0.5) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(title="Income vs Loan Amount by Approval",
       x="Annual Income", y="Loan Amount") +
  theme_minimal()

# CIBIL score boxplot
ggplot(df, aes(x = loan_status, y = cibil_score, fill = loan_status)) +
  geom_boxplot() +
  labs(title="CIBIL Score by Loan Status", x="Loan Status", y="CIBIL Score") +
  theme_minimal()

# Approval rate - Education
df %>% group_by(education) %>%
  summarise(rate = mean(loan_status=="Approved")) %>%
  ggplot(aes(x = education, y = rate, fill = education)) +
  geom_col() +
  labs(title="Approval Rate by Education", x="Education", y="Approval Rate") +
  theme_minimal()

# Approval rate - Employment Type
df %>% group_by(self_employed) %>%
  summarise(rate = mean(loan_status=="Approved")) %>%
  ggplot(aes(x = self_employed, y = rate, fill = self_employed)) +
  geom_col() +
  labs(title="Approval Rate by Self Employment", x="Self Employed", y="Approval Rate") +
  theme_minimal()

# Correlation heatmap
num_df <- df %>% select(income_annum, loan_amount, loan_term,
                        cibil_score, no_of_dependents, total_assets, loan_to_income)
cor_mat <- cor(num_df)
print(cor_mat)

melt(cor_mat) %>%
  ggplot(aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  labs(title="Correlation Heatmap") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  scale_fill_gradient2(midpoint = 0, limit = c(-1,1))
