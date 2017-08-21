library(ggplot2)
library(tibble)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)

tdates <- train %>% select(-Page)
foo <- train %>% select(Page) %>% rownames_to_column()
mediawiki <- foo %>% filter(str_detect(Page, "mediawiki"))
wikimedia <- foo %>% filter(str_detect(Page, "wikimedia"))
wikipedia <- foo %>% filter(str_detect(Page, "wikipedia")) %>% 
  filter(!str_detect(Page, "wikimedia")) %>%
  filter(!str_detect(Page, "mediawiki"))

wikipedia <- wikipedia %>%
  separate(Page, into = c("foo", "bar"), sep = ".wikipedia.org_") %>%
  separate(foo, into = c("article", "locale"), sep = -4) %>%
  separate(bar, into = c("access", "agent"), sep = "_") %>%
  mutate(locale = str_sub(locale,2,3))

wikimedia <- wikimedia %>%
  separate(Page, into = c("article", "bar"), sep = "_commons.wikimedia.org_") %>%
  separate(bar, into = c("access", "agent"), sep = "_") 
wikimedia$locale <- "wikmed"

mediawiki <- mediawiki %>%
  separate(Page, into = c("article", "bar"), sep = "_www.mediawiki.org_") %>%
  separate(bar, into = c("access", "agent"), sep = "_")
mediawiki$locale = "medwik"

tpages <- wikipedia %>%
  full_join(wikimedia, by = c("rowname", "article", "locale", "access", "agent")) %>%
  full_join(mediawiki, by = c("rowname", "article", "locale", "access", "agent"))