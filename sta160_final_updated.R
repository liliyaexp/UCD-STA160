library(tidyverse)
library(wordcloud)
library(tm)
library(glmnet)
library(plotmo)
library(pROC)

path_test = "~/Desktop/sta 160 dataset/test/test.csv"
path_train = "~/Desktop/sta 160 dataset/train.csv"
path_state_labels = "~/Desktop/sta 160 dataset/state_labels.csv"

train <- read_csv(path_train)
test <- read_csv(path_test)
train_text = train$Description

# Load the data as a corpus
docs <- Corpus(VectorSource(train_text))

# Text transformation

toSpace <-
  content_transformer(function (x , pattern)
    gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Text stemming
docs <- tm_map(docs, stemDocument)

##  Build a term-document matrix

dtm <- TermDocumentMatrix(docs)
dtm # show the sparsity

m <- as.matrix(dtm) # Convert the sparse matrix before use
v <- sort(rowSums(m), decreasing = TRUE)
d <- data.frame(word = names(v), freq = v)
head(d, 10)
tail(d, 10)

View(m)

## Generate the Word cloud

set.seed(1234)
wordcloud(
  words = d$word,
  freq = d$freq,
  min.freq = 1,
  max.words = 200,
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2")
)

terms = dtm$dimnames[[1]]
X = t(m)
Y = ifelse(train$AdoptionSpeed == 4, 0, 1)
Y = as.factor(Y)
y = Y

### fitting the lasso model

# using cross-validation to find the best lamdba

lambdas_to_try <- 10 ^ seq(-3, 5, length.out = 100)
cv.glmmod <- cv.glmnet(
  X,
  as.numeric(Y) - 1,
  alpha = 1,
  lambda = lambdas_to_try,
  nfolds = 10
)

# Plot cross-validation results
plot(cv.glmmod)
(best.lambda <- cv.glmmod$lambda.min) # 0.006428073

# show how the lambda affect the coefficients

res <-
  glmnet(X,
         y,
         alpha = 1,
         lambda = lambdas_to_try,
         standardize = FALSE)
plot_glmnet(res, label = 10)

# for each lamdba, how many non-zero coefficient
number_nonzero_per_lambda = apply(res$beta, 2, function(x)
  sum(x != 0))
lambdas_to_try[c(80, 82, 83, 86, 87, 88, 89)]
res$beta[, 80 + 1][which(res$beta[, 80 + 1] != 0)] # for s80, include one coefficient
res$beta[, 82 + 1][which(res$beta[, 82 + 1] != 0)] # for s82, include two coefficient
res$beta[, 83 + 1][which(res$beta[, 83 + 1] != 0)] # for s82, include 3 coefficient
res$beta[, 86 + 1][which(res$beta[, 86 + 1] != 0)] # for s86, include 6 coefficient
res$beta[, 87 + 1][which(res$beta[, 87 + 1] != 0)] # for s87, include 7 coefficient
res$beta[, 88 + 1][which(res$beta[, 88 + 1] != 0)] # for s88, include 9 coefficient
res$beta[, 89 + 1][which(res$beta[, 89 + 1] != 0)] # for s89, include 13 coefficient

# Fit final model and get its sum of squared residuals and multiple R-squared

y = as.numeric(Y) - 1
model_cv <-
  glmnet(X,
         y,
         alpha = 1,
         lambda = best.lambda,
         standardize = TRUE)
y_hat_cv <- predict(model_cv, X)
ssr_cv <- t(y - y_hat_cv) %*% (y - y_hat_cv)
rsq_lasso_cv <- cor(y, y_hat_cv) ^ 2

# find the best cut-off for prediction

cutoff = seq(0.1, 0.9, 0.05)
nn = length(cutoff)
mis_classified_rate = rep(0, nn)
for (i in 1:nn) {
  y_hat_label = y_hat_cv > cutoff[i]
  mis_classified_rate[i] = 1 - (sum(diag(table(y_hat_label, y))) / sum(table(y_hat_label, y)))
}

plot(
  cutoff,
  mis_classified_rate,
  type = 'b',
  xlab = "prediction cut-off",
  ylab = "misclassification rate",
  main = "Selecting the best cutoff for prediction"
)

min(mis_classified_rate)
best_cutoff = cutoff[which.min(mis_classified_rate)]

# apply the best cut off to get the prediction result, misclassification rate and the confusion matrix

y_hat_label = y_hat_cv > 0.65
table(y_hat_label, y) # confusion matrix
roc1 = roc(y_hat_label, y)
roc1$auc

# inspect the coefficients of the final model

glmmod_coef = coef(model_cv) # the first one is the intercept, therefore, the total number of coefficients = total number of terms + 1
sum(glmmod_coef != 0)

# the least 20 coefficients

lowest20_order = order(glmmod_coef)[1:20]
terms[lowest20_order]
glmmod_coef[lowest20_order]

# the highest 20 coefficients

high20_order = order(glmmod_coef, decreasing = TRUE)[1:20]
glmmod_coef[high20_order]
terms[high20_order]

low20_df = data.frame(term = terms[lowest20_order], coef = glmmod_coef[lowest20_order])
high20_df = data.frame(term = terms[high20_order], coef = glmmod_coef[high20_order])

##########
#### apply this on the test data set

test_text = test$Description

# Load the data as a corpus
test_docs <- Corpus(VectorSource(test_text))

# Text transformation

toSpace <-
  content_transformer(function (x , pattern)
    gsub(pattern, " ", x))
test_docs <- tm_map(test_docs, toSpace, "/")
test_docs <- tm_map(test_docs, toSpace, "@")
test_docs <- tm_map(test_docs, toSpace, "\\|")

# Convert the text to lower case
test_docs <- tm_map(test_docs, content_transformer(tolower))
# Remove numbers
test_docs <- tm_map(test_docs, removeNumbers)
# Remove english common stopwords
test_docs <- tm_map(test_docs, removeWords, stopwords("english"))
# Remove punctuations
test_docs <- tm_map(test_docs, removePunctuation)
# Eliminate extra white spaces
test_docs <- tm_map(test_docs, stripWhitespace)
# Text stemming
test_docs <- tm_map(test_docs, stemDocument)

##  Build a term-document matrix

test_dtm <- TermDocumentMatrix(test_docs)
test_dtm # show the sparsity

test_m <- as.matrix(test_dtm) # Convert the sparse matrix before use
test_v <- sort(rowSums(test_m), decreasing = TRUE)
test_d <- data.frame(word = names(test_v), freq = test_v)

# generate the word cloud for test data

wordcloud(
  words = test_d$word,
  freq = test_d$freq,
  min.freq = 1,
  max.words = 200,
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2")
)

test_terms = test_dtm$dimnames[[1]]
test_X = t(test_m)

# apply the final model on the test data

test_y_hat_cv <- predict(model_cv, test_X)
ssr_cv <- t(y - y_hat_cv) %*% (y - y_hat_cv)
rsq_lasso_cv <- cor(y, y_hat_cv) ^ 2

# display the number of observations per state
train_test %>%
  filter(df == "train") %>%
  ggplot(aes(x = StateName, fill = AdoptionSpeed)) +
  geom_bar(position = "fill", color = "black") +
  theme_minimal() +
  scale_y_continuous(labels = percent) +
  scale_fill_brewer(palette = "Purples") +
  coord_flip()

#Description Metadata analysis
train_test <- train_test %>%
  mutate(
    DescriptionCharacterLength = str_length(Description),
    DescriptionSentencesCount = str_count(Description, "[[:alnum:] ][.!?]"),
    DescriptionWordCount = str_count(Description, "[[:alpha:][-]]+"),
    DescriptionCapitalsCount = str_count(Description, "[A-Z]"),
    DescriptionLettersCount = str_count(Description, "[A-Za-z]"),
    DescriptionPunctuationCount = str_count(Description, "[[:punct:]]"),
    DescriptionExclamationCount = str_count(Description, fixed("!")),
    DescriptionQuestionCount = str_count(Description, fixed("?")),
    DescriptionDigitsCount = str_count(Description, "[[:digit:]]"),
    DescriptionDistinctWordsCount = lengths(lapply(strsplit(Description, split = ' '), unique)),
    DescriptionLexicalDensity = DescriptionDistinctWordsCount / DescriptionWordCount
  )

train_test %>%
  filter(!is.na(AdoptionSpeed)) %>%
  select(AdoptionSpeed, starts_with("Description"),-Description) %>%
  gather(key = "Variable", value = "value",-AdoptionSpeed) %>%
  ggplot(aes(x = log(value + 1), fill = AdoptionSpeed)) +
  geom_density(alpha = 0.4) +
  labs(title = "Description Metadata Analysis",
       subtitle = "All metrics are counts except for 'DescriptionLexicalDensity'",
       x = "Log transformed metric") +
  scale_fill_brewer(palette = "OrRd") +
  facet_wrap( ~ Variable, scales = "free") +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold"))