Population aging and COVID-19 death rate relationship analysis
================
Fanli Zhou
2020/5/15

  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Methods](#methods)
      - [Data](#data)
      - [Analysis Methods](#analysis-methods)
  - [Results and Discussion](#results-and-discussion)
      - [1. multiple linear regression](#multiple-linear-regression)
      - [2. Data transformation and GLM](#data-transformation-and-glm)
      - [3. Bootstrapping](#bootstrapping)
      - [Conclusions](#conclusions)
  - [Limitations](#limitations)
  - [References](#references)

# Summary

In this project, I attempted to use multiple linear regression and
general linear models to study the relationship between the portion of
the senior (age 65 and up) in the population and the COVID-19 death
rate. To approach this problem, I tested 11 potential confounding
variables and several statistical models. My final statistical model is
a GLM model including only two confounding variables, `med_bed` (the
number of hospital beds per 1,000 people) and `death_100_ind` (the ratio
of days with fewer than 100 accumulated deaths from 1/21/2020 to
5/14/2020). I assessed the model with bootstrapping and provided some
evidence that the COVID-19 death rate is positively associated with the
portion of the senior in the population.

# Introduction

The world is fighting with COVID-19. The COVID-19 is a new disease and
we are updating our knowledge about it every day. One well document
factor that leads to death from COVID-19 is age. As reported by the US
[CDC](https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/older-adults.html),
adults 65 years old and older have a high risk “for developing more
serious complications from COVID-19 illness” (cdc.gov 2020). In other
words, the COVID-19 death rate for the senior is higher than that for
the younger people. This conclusion is based on the COVID-19 deaths
statistics in the US.

To further support this conclusion, I decided to study the relationship
between age and the COVID-19 death rate with data science techniques.
Considering that COVID-19 patient information would not be public to
protect privacy, I would instead study the relationship between
population aging and the COVID-19 death rate. Here is my research
question:

**Does the portion of the senior (age 65 and up) in the population
associate with the COVID-19 death rate?**

  - Null hypothesis (![H\_0](https://latex.codecogs.com/png.latex?H_0
    "H_0")):
    
      - The portion of the senior (age 65 and up) in the population is
        not associated with the COVID-19 death rate.

  - The alternative hypothesis
    (![H\_A](https://latex.codecogs.com/png.latex?H_A "H_A")):
    
      - The portion of the senior (age 65 and up) in the population is
        associated with the COVID-19 death rate.

This study would help address the relationship between age and the
COVID-19 death rate. And hopefully, it would bring more attention to
protect the senior people from COVID-19 and protect ourselves.

# Methods

## Data

The two COVID-19 datasets, [the global confirmed
dataset](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv)
and [the global death
dataset](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv),
are from the [data
repository](https://github.com/CSSEGISandData/COVID-19) for the 2019
Novel Coronavirus Visual Dashboard operated by the Johns Hopkins
University Center for Systems Science and Engineering (JHU CSSE). The
dataset is updated every day starting from January 22nd, 2020. Both
datasets contain columns of location information (`Province/State`,
`Country/Region`, `Lat`, and `Long`) or daily accumulated numbers of
confirmed or deaths in the corresponding location, starting from
`1/22/20` to present (`5/14/2020`). As stated in the dataset repository,
“Australia, Canada and China are reported at the province/state level.
Dependencies of the Netherlands, the UK, France, and Denmark are listed
under the province/state level. The US and other countries are at the
country level.” The datasets combine data from multiple sources and may
contain mistakes.

Here is the information of variables used in this study:

| varaiable Name    | Min   | Max   | Description and Source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | ----- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rate`            | 0     | 0.16  | For each country, the death rate (by 5/14/2020) is calculated by ![\\frac{N\_{\\text{COVID-19 death}}}{N\_{\\text{COVID-19 confirmed}}}](https://latex.codecogs.com/png.latex?%5Cfrac%7BN_%7B%5Ctext%7BCOVID-19%20death%7D%7D%7D%7BN_%7B%5Ctext%7BCOVID-19%20confirmed%7D%7D%7D "\\frac{N_{\\text{COVID-19 death}}}{N_{\\text{COVID-19 confirmed}}}") with data from [global confirmed](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv) and [global death](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv) datasets                                                                                                                                        |
| `age_65up`        | 1.09  | 27.58 | [Population ages 65 and above (% of total population)](https://data.worldbank.org/indicator/SP.POP.65UP.TO.ZS) from the [World Bank Open Data](https://data.worldbank.org/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `air_pollution`   | 5.86  | 99.73 | [PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)](https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3) from the World Bank Open Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `med_bed`         | 0.1   | 13.4  | [Hospital beds (per 1,000 people)](https://data.worldbank.org/indicator/SH.MED.BEDS.ZS)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `physicians`      | 0     | 8.19  | [Physicians (per 1,000 people)](https://data.worldbank.org/indicator/SH.MED.PHYS.ZS) from the World Bank Open Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `nurses_midwives` | 0.06  | 18.12 | [Nurses and midwives (per 1,000 people)](https://data.worldbank.org/indicator/SH.MED.NUMW.P3) from the World Bank Open Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `pop_density`     | 2.04  | 7953  | [Population density (people per sq. km of land area)](https://data.worldbank.org/indicator/EN.POP.DNST) from the World Bank Open Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `smoking`         | 2     | 43.4  | [Smoking prevalence, total (ages 15+)](https://data.worldbank.org/indicator/SH.PRV.SMOK) from the World Bank Open Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `con_500_ind`     | 0     | 1     | The ratio of days with fewer than 500 confirmed cases (accumulated) is calculated by ![\\frac{N\_{\\text{days with 1-500 confirmed cases 1/21/2020-5/14/2020}}}{N\_{\\text{days with more than one confirmed cases 1/21/2020-5/14/2020}}}](https://latex.codecogs.com/png.latex?%5Cfrac%7BN_%7B%5Ctext%7Bdays%20with%201-500%20confirmed%20cases%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D%7BN_%7B%5Ctext%7Bdays%20with%20more%20than%20one%20confirmed%20cases%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D "\\frac{N_{\\text{days with 1-500 confirmed cases 1/21/2020-5/14/2020}}}{N_{\\text{days with more than one confirmed cases 1/21/2020-5/14/2020}}}") with data from the [global confirmed](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv) dataset |
| `death_100_ind`   | 0     | 1     | The ratio of days with fewer than 100 deaths (accumulated) is calculated by ![\\frac{N\_{\\text{days with 1-100 deaths 1/21/2020-5/14/2020}}}{N\_{\\text{days with more than one deaths since 1/21/2020-5/14/2020}}}](https://latex.codecogs.com/png.latex?%5Cfrac%7BN_%7B%5Ctext%7Bdays%20with%201-100%20deaths%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D%7BN_%7B%5Ctext%7Bdays%20with%20more%20than%20one%20deaths%20since%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D "\\frac{N_{\\text{days with 1-100 deaths 1/21/2020-5/14/2020}}}{N_{\\text{days with more than one deaths since 1/21/2020-5/14/2020}}}") with data from the [global death](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv) dataset                                                       |
| `confirmed_rate`  | 0     | 0.04  | The confirmed rate (by 5/14/2020) is calculated by ![\\frac{N\_{\\text{COVID-19 confirmed}}}{N\_{\\text{population}}}](https://latex.codecogs.com/png.latex?%5Cfrac%7BN_%7B%5Ctext%7BCOVID-19%20confirmed%7D%7D%7D%7BN_%7B%5Ctext%7Bpopulation%7D%7D%7D "\\frac{N_{\\text{COVID-19 confirmed}}}{N_{\\text{population}}}") with data from the [global confirmed](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv) dataset and the [population, total](https://data.worldbank.org/indicator/SP.POP.TOTL) dataset from the World Bank Open Data                                                                                                                                                                                                           |
| `gov_resp`        | 18.25 | 100   | This value is the highest stringency index (by 5/14/2020) for each country from the [Oxford Covid-19 Government Response Tracker data](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv) from the [Oxford Covid-19 Government Response Tracker repository](https://github.com/OxCGRT/covid-policy-tracker/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `ind_80`          | 0.15  | 1     | This index is calculated by ![\\frac{N\_{\\text{days with 0-80 stringency index 1/21/2020-5/14/2020}}}{N\_{\\text{days 1/21/2020-5/14/2020}}}](https://latex.codecogs.com/png.latex?%5Cfrac%7BN_%7B%5Ctext%7Bdays%20with%200-80%20stringency%20index%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D%7BN_%7B%5Ctext%7Bdays%201%2F21%2F2020-5%2F14%2F2020%7D%7D%7D "\\frac{N_{\\text{days with 0-80 stringency index 1/21/2020-5/14/2020}}}{N_{\\text{days 1/21/2020-5/14/2020}}}") with data from the [Oxford Covid-19 Government Response Tracker data](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv)                                                                                                                                                                                                        |

## Analysis Methods

The multiple linear regression, GLM (General Linear Model), and
bootstrapping algorithms or methods were used to study the relationship
between the COVID-19 death rate and the portion of the senior. The
Python programming language (Van Rossum and Drake 2009), the R
programming language (R Core Team 2019) and the following Python/R
packages were used: Pandas (McKinney 2010), requests (Chandra and
Varanasi 2015), knitr (Xie 2014), tidyverse (Wickham 2017), MASS
(Venables and Ripley 2002), mice (van Buuren and Groothuis-Oudshoorn
2011), resample (Hesterberg 2015), testthat (Wickham 2011), rjson
(Couture-Beil 2018), repr (Angerer, Kluyver, and Schulz 2019). The code
used to perform the analysis and create this report can be found here:
<https://github.com/flizhou/age_effects_on_COVID-19>.

The following three functions were used to display model results and
analysis.

``` r
#' Report linear regression results
#'
#' @param model the fitted model to analyze
#'
#' @return results dataframe
analyze_lm <- function(model){
    model %>%
        broom::tidy() %>%
        mutate(p.value.adjust = p.adjust(p.value, method = "BH"),
               significant = p.value.adjust < 0.05)
}

#' Plots linear regression diagnostics results
#'
#' @param model the fitted model to analyze
#' @param formula the model formula
#'
#' @return plots
plot_model_diag <- function(model, formula = "") {
    options(repr.plot.width = 20, repr.plot.height = 20)
    par(mfrow = c(2, 2), mar = c(5, 5, 4, 2) + 0.1, 
        cex.lab = 1.2, cex.axis = 1, cex.main = 1.2, cex.sub = 1.2)

    if (formula != "") {
        print(paste("Formula =", formula))
    }
    
    plot(model, cex.caption = 1.2, cex.id = 1)
}

t <- theme(plot.title = element_text(size = 16),
           axis.title = element_text(size = 14),
           text = element_text(size = 12),
           legend.title = element_text(size = 14),
           legend.text = element_text(size = 14),
           strip.text.x = element_text(size = 14))

#' Plots bootstrapping results
#'
#' @param boot_results the bootstrapping results
#' @param var the variable to plot
#' @param model_results the model results
#'
#' @return ggplot plot
plot_results <-  function (boot_results, var, model_results){
    
    boot_fits <- boot_results %>% 
        filter(term == var)
    
    obs <- model_results %>% 
        filter(term == var)
    
    t_star <- (boot_fits$estimate - obs$estimate) / boot_fits$std.error
    pval_boot <- round((1 + sum(abs(t_star) > abs(obs$statistic))) / (N + 1), 4)

    df <- tibble(estimate = boot_fits$estimate,
           t_star = t_star) %>%
        gather("term", "value") 
    
    df %>%
        ggplot() +
            geom_histogram(aes(value), bins = 30) +
            facet_grid(cols = vars(term), scales = "free") + 
            geom_vline(data = df %>% filter(term == "t_star"), 
                       aes(xintercept = obs$statistic, color = "Observed t")) +
            geom_vline(data = df %>% filter(term == "estimate"), 
                       aes(xintercept = obs$estimate, color = "Observed estimate")) +
            theme_bw() +
            labs(title = paste("Bootstrap distributions of the coefficient of ", var, 
                               "\np-value = ", pval_boot, sep = ""),
                 color = "Line") + t
    
}
```

# Results and Discussion

## 1\. multiple linear regression

I started my analysis with a multiple linear regression model to study
the relationship between the COVID-19 death rate
(![Y](https://latex.codecogs.com/png.latex?Y "Y"), rate) and the portion
of the senior (![X](https://latex.codecogs.com/png.latex?X "X"),
age\_65up). A multiple linear regression model may not give me the most
accurate model but is highly interpretable. To answer my research
questions, I would sacrifice some accuracy for interpretability.

I first tried the full model, which includes all confounding variables,
and picked out potential useful confounding variables based on
statistical significance (p-value \< 0.05). Then I tested reduced models
to decide which variables to include in the final model. Considering
that I was doing multiple testing, I controlled for multiple comparisons
using Benjamini & Hochberg (BH) False Discovery Rate (FDR) adjustment
(setting FDR to 5%). A detailed analysis process is in my [EDA
notebook](https://github.com/flizhou/age_effects_on_COVID-19/blob/master/scripts/eda.ipynb).
Based on these analyses, my final model only includes `med_bed`,
`death_100_ind`, and `confirmed_rate` as confounding variables.

``` r
# load the complete dataset that includes all confounding variables
data <- read_csv("../data/clean_data/complete_data.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   country = col_character(),
    ##   age_65up = col_double(),
    ##   air_pollution = col_double(),
    ##   med_bed = col_double(),
    ##   physicians = col_double(),
    ##   nurses_midwives = col_double(),
    ##   pop_density = col_double(),
    ##   smoking = col_double(),
    ##   con_500_ind = col_double(),
    ##   death_100_ind = col_double(),
    ##   rate = col_double(),
    ##   confirmed_rate = col_double(),
    ##   ind_80 = col_double(),
    ##   gov_resp = col_double()
    ## )

``` r
# remove confounding variables that won't be used in the final model
df_sub <- data %>%
    select(country, rate, age_65up, med_bed, 
           death_100_ind, confirmed_rate) %>%
    drop_na()
head(df_sub)
```

    ## # A tibble: 6 x 6
    ##   country       rate age_65up med_bed death_100_ind confirmed_rate
    ##   <chr>        <dbl>    <dbl>   <dbl>         <dbl>          <dbl>
    ## 1 Afghanistan 0.0241     2.58     0.5         0.889     0.000152  
    ## 2 Albania     0.0345    13.7      2.9         1         0.000313  
    ## 3 Algeria     0.0821     6.36     1.9         0.475     0.000153  
    ## 4 Angola      0.0417     2.22     0.8         1         0.00000156
    ## 5 Argentina   0.0495    11.1      5           0.575     0.000160  
    ## 6 Australia   0.0140    15.7      3.8         0.545     0.00225

``` r
# the final multiple linear regression model
model_1 <- lm(rate ~ age_65up + med_bed + death_100_ind + confirmed_rate, df_sub[-1])

analyze_lm(model_1)
```

    ## # A tibble: 5 x 7
    ##   term         estimate std.error statistic   p.value p.value.adjust significant
    ##   <chr>           <dbl>     <dbl>     <dbl>     <dbl>          <dbl> <lgl>      
    ## 1 (Intercept)   0.0639   0.0130        4.93   2.32e-6      0.0000116 TRUE       
    ## 2 age_65up      0.00158  0.000622      2.53   1.25e-2      0.0156    TRUE       
    ## 3 med_bed      -0.00333  0.00151      -2.20   2.91e-2      0.0291    TRUE       
    ## 4 death_100_i~ -0.0383   0.0124       -3.10   2.35e-3      0.00588   TRUE       
    ## 5 confirmed_r~  1.65     0.596         2.77   6.28e-3      0.0105    TRUE

To assess this model, I checked whether the “LINE” conditions for
multiple linear regression hold. The data were collected in each country
independently, so I would assume that the independence assumption is
reasonable. Then, let’s look at the rest assumptions:

``` r
# plot model diagnostics
plot_model_diag(model_1, "rate ~ age_65up + med_bed + death_100_ind + confirmed_rate")
```

    ## [1] "Formula = rate ~ age_65up + med_bed + death_100_ind + confirmed_rate"

![](report_files/figure-gfm/model%201%20analysis-1.png)<!-- -->

``` r
# Compute Shapiro-Wilk test of normality
rstatix::shapiro_test(residuals(model_1))
```

    ## Registered S3 methods overwritten by 'car':
    ##   method                          from
    ##   influence.merMod                lme4
    ##   cooks.distance.influence.merMod lme4
    ##   dfbeta.influence.merMod         lme4
    ##   dfbetas.influence.merMod        lme4

    ## # A tibble: 1 x 3
    ##   variable           statistic      p.value
    ##   <chr>                  <dbl>        <dbl>
    ## 1 residuals(model_1)     0.903 0.0000000315

``` r
# Compute Bartlett test of homogeneity of variances
df_sub[-1] %>%
    bartlett.test(rate ~ age_65up + med_bed + death_100_ind + confirmed_rate)
```

    ## 
    ##  Bartlett test of homogeneity of variances
    ## 
    ## data:  .
    ## Bartlett's K-squared = 3706.3, df = 4, p-value < 2.2e-16

First, check the Residuals vs Fitted plot for the linearity assumption.
Residuals should have zero expectations at each fitted value. There are
some obvious patterns in the plot and the averages of residuals are away
from zero. So the linearity assumption does not hold and the LSE (Least
Squares Estimator) might be biased and even inconsistent.

Second, check the Normal Q-Q plot for the normality assumption.
Deviations of the lower and upper tails from the straight line indicates
that the distribution of residuals does not conform to a normal
distribution (right-skewed in this case). Besides, the Shapiro-Wilk test
of normality is significant. So the normality assumption is violated.

Third, check the Scale-Location plot for the equal variance assumption.
There is an overall positive trend in the plot so the variance is not
constant. Besides, the Bartlett test of homogeneity of variances is
significant. So the equal variance assumption does not hold and the LSE
may not be the BLUE (Best Linear Unbiased Estimator).

Finally, check the Residuals vs. Leverage plot for influential points.
Three points are labeled for high Cook’s distance values.

The model violates most of the “LINE” conditions. I next considered
improving the model with data transformation and GLM (General Linear
Model).

## 2\. Data transformation and GLM

I tried to improve the model by trial and error. Here is the final model
I got. The final model was the best identified, but not necessarily the
best model.

``` r
model_2 <- df_sub %>%
    mutate(rate = (rate*(133 -1) + 2)/133) %>%
    glm(rate ~ log(age_65up) + log(med_bed) + 
        log(death_100_ind + 1) + log(confirmed_rate), 
        family = Gamma(link = "identity"), .)

analyze_lm(model_2)
```

    ## # A tibble: 5 x 7
    ##   term          estimate std.error statistic  p.value p.value.adjust significant
    ##   <chr>            <dbl>     <dbl>     <dbl>    <dbl>          <dbl> <lgl>      
    ## 1 (Intercept)    0.0748   0.0178        4.21  4.51e-5      0.000113  TRUE       
    ## 2 log(age_65up)  0.0138   0.00355       3.88  1.63e-4      0.000272  TRUE       
    ## 3 log(med_bed)  -0.0102   0.00334      -3.06  2.67e-3      0.00267   TRUE       
    ## 4 log(death_10~ -0.113    0.0245       -4.61  8.91e-6      0.0000446 TRUE       
    ## 5 log(confirme~ -0.00312  0.000981     -3.18  1.79e-3      0.00223   TRUE

Again I checked whether the “LINE” conditions hold to assess the model.
The independence assumption is reasonable as before. For the rest
assumptions:

``` r
# plot model diagnostics
plot_model_diag(model_2)
```

![](report_files/figure-gfm/model%202%20analysis-1.png)<!-- -->

``` r
# Compute Shapiro-Wilk test of normality
rstatix::shapiro_test(residuals(model_2))
```

    ## # A tibble: 1 x 3
    ##   variable           statistic p.value
    ##   <chr>                  <dbl>   <dbl>
    ## 1 residuals(model_2)     0.985   0.120

``` r
# Compute Bartlett test of homogeneity of variances
df_sub[-1] %>%
    mutate(rate = (rate*(133 -1) + 2)/133) %>%
    bartlett.test(rate ~ log(age_65up) + log(med_bed) + log(death_100_ind+1) + log(confirmed_rate))
```

    ## 
    ##  Bartlett test of homogeneity of variances
    ## 
    ## data:  .
    ## Bartlett's K-squared = 3708.5, df = 4, p-value < 2.2e-16

First, the averages of residuals are slightly smaller than zero in the
Residuals vs Fitted plot but may not be a big problem. I would say that
the linearity assumption could hold here.

Second, the lower and upper tails are slightly away from the straight
line in the Normal Q-Q plot. Besides, the Shapiro-Wilk test of normality
is significant but the p-value is very close to 0.05. So there is some
evidence that the normality assumption could be reasonable.

Third, the Scale-Location plot has some trends so the variance is not
constant. And the Bartlett test of homogeneity of variances is
significant. So the normality assumption could not hold. Finally, three
points are labeled for high Cook’s distance values in the Residuals
vs. Leverage plot.

The final model still violates some of the “LINE” conditions. So it may
not be reasonable to use the theoretical inferential conclusions.
Instead, I used bootstrapping as an alternative approach to statistical
inference.

## 3\. Bootstrapping

I took bootstrap samples from 133 countries in `df_sub` with
replacement. This was case resampling, so the information from each
country remained together. Here I had to drop the `log(confirmed_rate)`
term in the formula to make the `glm` results converge.

``` r
set.seed(12345)
N <- 1000

model_3 <- df_sub %>%
    mutate(rate = (rate*(133 -1) + 2)/133) %>%
    glm(rate ~ log(age_65up) + log(med_bed) + 
        log(death_100_ind + 1), 
        family = Gamma(link = "identity"), .)

model_3_results <- analyze_lm(model_3)

model_3_results
```

    ## # A tibble: 4 x 7
    ##   term           estimate std.error statistic p.value p.value.adjust significant
    ##   <chr>             <dbl>     <dbl>     <dbl>   <dbl>          <dbl> <lgl>      
    ## 1 (Intercept)      0.0850   0.0197       4.31 3.01e-5       0.000121 TRUE       
    ## 2 log(age_65up)    0.0144   0.00428      3.37 9.82e-4       0.00113  TRUE       
    ## 3 log(med_bed)    -0.0137   0.00338     -4.04 8.68e-5       0.000174 TRUE       
    ## 4 log(death_100~  -0.0829   0.0249      -3.32 1.13e-3       0.00113  TRUE

``` r
boot_results <- df_sub %>%
    mutate(rate = (rate*(133 -1) + 2)/133) %>%
    rsample::bootstraps(N) %>% 
    mutate(lm = map(splits, ~ glm(rate ~ log(age_65up) + log(med_bed) + 
                                  log(death_100_ind + 1), 
                                  family = Gamma(link = "identity"), .)),
           tidy = map(lm, broom::tidy)) %>% 
    select(-splits, -lm) %>% 
    unnest(tidy)

plot_results(boot_results, "log(age_65up)", model_3_results)
```

![](report_files/figure-gfm/model%203-1.png)<!-- -->

``` r
plot_results(boot_results, "log(med_bed)", model_3_results)
```

![](report_files/figure-gfm/model%203-2.png)<!-- -->

``` r
plot_results(boot_results, "log(death_100_ind + 1)", model_3_results)
```

![](report_files/figure-gfm/model%203-3.png)<!-- -->

Based on the bootstrapping results, the coefficients of `log(age_65up)`,
`log(med_bed)`, `log(death_100_ind + 1)` are all significant (p-value \<
0.05).

## Conclusions

In this analysis, I attempted to use a linear model to explain the
relationship between the portion of the senior (`age_65up`%) in the
population and the COVID-19 death rate. I tried several linear models
and finally decided to train a generalized linear model with a
Gamma-distribution dependent variable. My final model is:

![\\frac{132\\text{rate}+2}{133}](https://latex.codecogs.com/png.latex?%5Cfrac%7B132%5Ctext%7Brate%7D%2B2%7D%7B133%7D
"\\frac{132\\text{rate}+2}{133}")
![=\\0.0850+\\0.0144\\text{log(age\\\_65up)}-\\0.0137\\text{log(med\\\_bed)}-\\0.0829\\text{log(death\\\_100\\\_ind+1)}](https://latex.codecogs.com/png.latex?%3D%5C0.0850%2B%5C0.0144%5Ctext%7Blog%28age%5C_65up%29%7D-%5C0.0137%5Ctext%7Blog%28med%5C_bed%29%7D-%5C0.0829%5Ctext%7Blog%28death%5C_100%5C_ind%2B1%29%7D
"=\\0.0850+\\0.0144\\text{log(age\\_65up)}-\\0.0137\\text{log(med\\_bed)}-\\0.0829\\text{log(death\\_100\\_ind+1)}")

where the response (rate) follows a Gamma distribution. I understand
that this is a simplified model and I may not include all possible
confounding variables in the model. And the “LINE” conditions for
multiple linear models may not hold for this model. So I used
bootstrapping for statistical inference. Given the bootstrapping
results, the coefficient of `log(age_65up)` is significant (p-value \<
0.05). I have the evidence to reject the null hypothesis and accept the
alternative hypothesis. So the portion of the senior (age 65 and up) in
the population is positively associated with the COVID-19 death rate.

# Limitations

My analysis is based on observational datasets. Some country statistics
data used in this analysis are not up-to-date. And the government
response index, the stringency index from the [Oxford Covid-19
Government Response Tracker
data](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv),
is subjective and may not reflect how effective the government response
was. Another big limitation that I could not include all possible
confounding variables in this analysis and I may have ignored some
important confounding variables. The best way to analyze the
relationship between age and the COVID-19 death rate is by conducting
well-designed experiments that control confounding variables.

# References

<div id="refs" class="references">

<div id="ref-repr">

Angerer, Philipp, Thomas Kluyver, and Jan Schulz. 2019. *Repr:
Serializable Representations*.
<https://CRAN.R-project.org/package=repr>.

</div>

<div id="ref-cdc">

cdc.gov. 2020. “Older Adults.” CDC.
<https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/older-adults.html>.

</div>

<div id="ref-chandra2015python">

Chandra, Rakesh Vidya, and Bala Subrahmanyam Varanasi. 2015. *Python
Requests Essentials*. Packt Publishing Ltd.

</div>

<div id="ref-rjson">

Couture-Beil, Alex. 2018. *Rjson: JSON for R*.
<https://CRAN.R-project.org/package=rjson>.

</div>

<div id="ref-resample">

Hesterberg, Tim. 2015. *Resample: Resampling Functions*.
<https://CRAN.R-project.org/package=resample>.

</div>

<div id="ref-mckinney-proc-scipy-2010">

McKinney, Wes. 2010. “Data Structures for Statistical Computing in
Python.” In *Proceedings of the 9th Python in Science Conference*,
edited by Stéfan van der Walt and Jarrod Millman, 51–56.

</div>

<div id="ref-R">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-mice">

van Buuren, Stef, and Karin Groothuis-Oudshoorn. 2011. “mice:
Multivariate Imputation by Chained Equations in R.” *Journal of
Statistical Software* 45 (3): 1–67.
<https://www.jstatsoft.org/v45/i03/>.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-mass">

Venables, W. N., and B. D. Ripley. 2002. *Modern Applied Statistics with
S*. Fourth. New York: Springer. <http://www.stats.ox.ac.uk/pub/MASS4>.

</div>

<div id="ref-testthat">

Wickham, Hadley. 2011. “Testthat: Get Started with Testing.” *The R
Journal* 3: 5–10.
<https://journal.r-project.org/archive/2011-1/RJournal_2011-1_Wickham.pdf>.

</div>

<div id="ref-tidyverse">

———. 2017. *Tidyverse: Easily Install and Load the ’Tidyverse’*.
<https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
