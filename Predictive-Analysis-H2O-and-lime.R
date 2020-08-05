

#install.packages(c("tidyverse", "h2o", "lime", "recipes"))



# Task 2: Import Libraries
library(tidyverse)
library(h2o)
library(lime)
library(recipes)



# Task 2.1: Load the IBM Employee Attrition Data
hr_data_raw <- read_csv("HR-Employee-Attrition.csv")
hr_data_raw[1:10,]




# Task 3: Pre-process Data
hr_data <- hr_data_raw %>%
    mutate_if(is.character, as.factor) %>%
    select(Attrition, everything())

recipe_data <- hr_data %>%
    recipe(formula = Attrition ~ .) %>%
    step_rm(EmployeeNumber) %>%
    step_zv(all_predictors()) %>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    prep(data = hr_data)

hr_data <- bake(recipe_data, new_data = hr_data) 
glimpse(hr_data)




# Task 4.0: Start H2O Cluster and Create Train/Test Splits
h2o.init(max_mem_size = "4g")



# Task 4.1: Create Training and Test Sets
set.seed(1234)
hr_data_h2o <- as.h2o(hr_data)

splits <- h2o.splitFrame(hr_data_h2o, c(0.7, 0.15), seed = 1234)

train <- h2o.assign(splits[[1]], "train" )
valid <- h2o.assign(splits[[2]], "valid" )
test  <- h2o.assign(splits[[3]], "test" )



# Task 5: Run AutoML to Train and Tune Models

y <- "Attrition"
x <- setdiff(names(train), y)

aml <- h2o.automl(x = x, 
                  y = y,
                  training_frame = train,
                  leaderboard_frame = valid,
                  #max_runtime_secs  = 300,
                  max_runtime_secs = 60)


# Task 6: Leaderboard Exploration
lb <- aml@leaderboard
print(lb, n=nrow(lb))
best_model <- aml@leader
#model_ids <-as.data.frame(aml@leaderboard$model_id)[,1]
#best_model <- h2o.getModel(grep("StackedEnsemble_BestOfFamily", model_ids, value=TRUE)[1])

# Task 7: Model Performance Evaluation
perf <- h2o.performance(best_model, newdata = test)
optimal_threshold <- h2o.find_threshold_by_max_metric(perf, "f1")
metrics <- as.data.frame(h2o.metric(perf, optimal_threshold))
t(metrics)


# Task 8: Baselearner Variable Importance 
explainer <- lime(as.data.frame(train[,-31]), best_model, bin_continuous=FALSE)
explanation <- explain(as.data.frame(test[3:10, -31]),
                       explainer = explainer,
                       kernel_width = 1,
                       n_features = 5, 
                       n_labels = 1)
plot_features(explanation)
plot_explanations(explanation)


