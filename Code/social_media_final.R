
library(rtweet)
library(tidyverse)
library(lubridate)
library(openxlsx)
library(readxl)

setwd("~/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Final Project_Social_Media")

####### Get data ------

# anti_immigration <- search_tweets(
#   "anti immigration", n = 30000,retryonratelimit = TRUE
# )



airpower2 <- Map(
  "search_tweets",
  c("airpower AND apple", "airpower AND airpods"),
  n = 30000,retryonratelimit=TRUE,lang="en"
)
airpower_en=bind_rows(airpower2) %>% distinct()

write.xlsx(airpower_en,"airpower_en_full.xlsx")


names(anti_immigration)

he=anti_immigration%>% head() 


###### Clean data -----


## keep only the columns required

airpower_small = airpower_en %>% select("created_at","user_id" ,"screen_name","text", starts_with("reply_to"),ends_with("name"),contains("location"),ends_with("coords"),"mentions_user_id","mentions_screen_name","retweet_user_id","retweet_screen_name","retweet_followers_count","retweet_friends_count","retweet_statuses_count","followers_count","friends_count","listed_count","statuses_count","favourites_count","hashtags") 

airpower_small$hashtags_clean=NA
for (i in 1:length(airpower_small$hashtags)){
  airpower_small$hashtags_clean[i]= paste0(as.vector(sapply(airpower_small$hashtags[i],as.character)), collapse = ",")[1]
}


#write.xlsx(airpower_small,"airpower_en_v1.xlsx")
airpower_small <- read_excel("~/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Final Project_Social_Media/airpower_en_v1.xlsx", 
                                              col_types = c("date", "text", "text", 
                                                            "text", "text", "text", "text", "text", 
                                                            "text", "text", "text", "text", "text", 
                                                            "text", "text", "text", "text", "text", 
                                                            "text", "text", "text", "text", "text", 
                                                            "numeric", "numeric", "numeric", 
                                                            "numeric", "numeric", "numeric", 
                                                            "numeric", "numeric", "text", "text"))

# replies table ------

airpower_small$created_at =as.Date(airpower_small$created_at) 
airpower_small=airpower_small %>% filter(created_at>=as.Date("2019-03-29"),created_at<=as.Date("2019-04-02"))
replies_table=airpower_small %>% filter(!is.na(reply_to_screen_name))

required_replies=replies_table %>% select(screen_name,reply_to_screen_name)
required_replies["type"]="reply"
names(required_replies)[1]="column1"
names(required_replies)[2]="column2"


# mentions table -----
mentions_table=airpower_small %>% filter(!is.na(mentions_screen_name),mentions_screen_name!="NA")
required_mentions = mentions_table %>% select(screen_name,mentions_screen_name) 


str(required_mentions$mentions_screen_name)


# Clean the mentions_screen_name column

required_mentions$mentions_screen_name_clean = NA

for (i in 1:length(required_mentions$mentions_screen_name)){
  required_mentions$mentions_screen_name_clean[i]= paste0(as.vector(sapply(required_mentions$mentions_screen_name[i],as.character)), collapse = ",")[1]
}


required_mentions_actual=required_mentions %>%
  transform(mentions_screen_name_clean = strsplit(mentions_screen_name_clean, ",")) %>%
  unnest(mentions_screen_name_clean) %>% select(-mentions_screen_name)

required_mentions_actual['type']="mention"
names(required_mentions_actual)
names(required_mentions_actual)[1]="column1"
names(required_mentions_actual)[2]="column2"



# retweets table ------

retweets_table=airpower_small %>% filter(!is.na(retweet_screen_name))
required_retweets = retweets_table %>% select(screen_name,retweet_screen_name)

names(required_retweets)[1]="column1"
names(required_retweets)[2]="column2"
required_retweets['type']="retweet"


## Normal tweets table -----

tweets_table=airpower_small %>% filter(is.na(reply_to_screen_name),is.na(mentions_screen_name),is.na(retweet_screen_name))
required_tweets=tweets_table %>% select(screen_name)

required_tweets_actual=required_tweets %>% mutate(column2=screen_name,type="tweet")
names(required_tweets_actual)[1]="column1"


### Combine all tables to get one unified table -----

combined_table=rbind.data.frame(required_tweets_actual,required_retweets,required_mentions_actual,required_replies)

combined_table=combined_table %>% distinct()

# convert into XLSX file 

write.xlsx(combined_table,"till_2.xlsx")

## retweets received

temp=combined_table %>% filter(type=="retweet")
retweets_received=temp %>% count(column2) %>% arrange(desc(n)) 

names(retweets_received)[1]="screen_name"
names(retweets_received)[2]="retweets_received"

## mentions received

temp=combined_table %>% filter(type=="mention")
mentions_received=temp %>% count(column2) %>% arrange(desc(n)) 

names(mentions_received)[1]="screen_name"
names(mentions_received)[2]="mentions_received"



### Get all metrics for all users in the dataset ------
names(airpower)
airpower_small
names(airpower_small)

users1=airpower_small %>% select("user_id","screen_name","name","followers_count","friends_count","listed_count" ,"statuses_count","favourites_count","location" )%>% distinct()
names(users1)
users_retweeted=airpower_small %>% select("retweet_user_id","retweet_screen_name","retweet_name","retweet_followers_count","retweet_friends_count", "retweet_statuses_count","retweet_location" )%>% distinct()
names(users_retweeted) = c("user_id" ,"screen_name","name", "followers_count", "friends_count", "statuses_count","location")


users_quoted=airpower_en %>% select("quoted_user_id","quoted_screen_name","quoted_name","quoted_followers_count" ,"quoted_friends_count" ,  "quoted_statuses_count","quoted_location" )%>% distinct()
names(users_quoted)
names(users_quoted) = c("user_id" ,"screen_name","name", "followers_count", "friends_count", "statuses_count","location")


# Join both tables

all_users_info=users1 %>% full_join(users_retweeted)%>% full_join(users_quoted) %>% distinct()


## combine that with the other metrics

all_users_info=all_users_info %>% full_join(retweets_received) %>% full_join(mentions_received)
all_users_info %>% filter(!is.na(retweets_received))
all_users_info %>% filter(!is.na(mentions_received))

write.xlsx(all_users_info,"all_users_info.xlsx")

length(airpower$location[!is.na(airpower$location)& (airpower$location!="")])

### join to get the userids of the following


all_users_userids=all_users_info %>% select(user_id,screen_name) %>% distinct()

temp1=combined_table %>% left_join(all_users_userids,by=c("column1"="screen_name"))
names(temp1)[4] ="user_id_column1" 

temp1=temp1 %>% left_join(all_users_userids,by=c("column2"="screen_name"))
names(temp1)[5] ="user_id_column2" 

write.xlsx(temp1,"clean_airpower_userids.xlsx")
temp1 %>% select(user_id_column1,user_id_column2)



###
names(all_users_info)
all_users_info=read.xlsx("all_users_info.xlsx")
all_users_info_remove_dup=all_users_info %>% group_by(user_id,screen_name,name,location) %>% summarise(followers_count = min(followers_count,na.rm = TRUE),friends_count = min(friends_count,na.rm = TRUE),listed_count = min(listed_count,na.rm = TRUE),statuses_count = min(statuses_count,na.rm = TRUE),favourites_count = min(favourites_count,na.rm = TRUE),retweets_received = min(retweets_received,na.rm = TRUE),mentions_received = min(mentions_received,na.rm = TRUE))%>% ungroup()

all_users_info_remove_dup=all_users_info_remove_dup %>% group_by(screen_name) %>%mutate(rank=dense_rank(desc(listed_count))) %>% filter(rank==1)


####

all_users_info_remove_dup %>% count(screen_name) %>% arrange(desc(n))

t=all_users_info_remove_dup %>% filter(screen_name=="CNN")

write.xlsx(all_users_info_remove_dup%>%select(-rank),"all_users_info_remove_dup.xlsx")


### Change albert's file

airpower_en_v3$`Relationship Time` = as.Date(lubridate ::mdy_hm(airpower_en_v3$`Relationship Time`))

airpower_en_v3$`Relationship Time`=as.integer(airpower_en_v3$`Relationship Time`)


as.integer(as.Date("2019-03-29"))
as.integer(as.Date("2019-03-30"))
as.integer(as.Date("2019-03-29"))
as.integer(airpower_en_v3$`Relationship Time`[1])
as.integer(airpower_en_v3$`Relationship Time`[1])
as.integer(airpower_en_v3$`Relationship Time`[1])

write.csv(airpower_en_v3,"airpower_en_v5.csv")


######## Rate of growth of network #####

airpower_en_v4 = read_csv("airpower_en_v4.csv")
airpower_en_v4$`Relationship Time` = lubridate::dmy(airpower_en_v4$`Relationship Time`)

## March 29th
temp1=airpower_en_v4%>% filter(`Relationship Time`>="2019-03-25",`Relationship Time`<="2019-04-03") 
Source_unique=temp1$Source %>% unique()
Target_unique=temp1$Target %>% unique()
nodes_unique=c(Source_unique,Target_unique)%>% unique()%>% length()

edges_unique=temp1 %>% select(Source,Target) %>% distinct()%>%dim()%>%.[1]
