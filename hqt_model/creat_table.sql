create table if not exists tmp.hqt_train
(
aid bigint,
uid bigint,
label bigint
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
stored as textfile;

load data local inpath '/home/heqt/tencent/train.csv' overwrite into table tmp.hqt_train;


create table if not exists tmp.hqt_test1
(
aid bigint,
uid bigint
)
row format DELIMITED
fields TERMINATED by ','
stored as textfile;

load data local inpath '/home/heqt/tencent/test1.csv' overwrite into table tmp.hqt_test1;

create table if not exists tmp.hqt_adFeature
(
aid bigint,
advertiserId string,
campaignId string,
creativeId string,
creativeSize string,
adCategoryId string,
productId string,
productType string
)
row format DELIMITED
fields TERMINATED by ','
stored as textfile;

load data local inpath '/home/heqt/tencent/adFeature.csv' overwrite into table tmp.hqt_adFeature;

create table if not exists tmp.hqt_userFeature_2
(
uid bigint,
age string,
gender string,
marriageStatus string,
education string,
consumptionAbility string,
LBS string,
interest1 string,
interest2 string,
interest3 string,
interest4 string,
interest5 string,
kw1 string,
kw2 string,
kw3 string,
topic1 string,
topic2 string,
topic3 string,
appIdInstall string,
appIdAction string,
ct string,
os string,
carrier string,
house string
)
row format DELIMITED
fields terminated by ','
stored as textfile;
load data local inpath '/home/heqt/tencent/userFeature.csv' overwrite into table tmp.hqt_userFeature_2;




create table tmp.hqt_userFeature
as 
select uid,
case when age ='' then null else age end,
case when gender ='' then null else gender end,
case when marriageStatus ='' then null else marriageStatus end,
case when education ='' then null else education end,
case when consumptionAbility ='' then null else consumptionAbility end,
case when LBS ='' then null else LBS end,
case when interest1 ='' then null else interest1 end,
case when interest2 ='' then null else interest2 end,
case when interest3 ='' then null else interest3 end,
case when interest4 ='' then null else interest4 end,
case when interest5 ='' then null else interest5 end,
case when kw1 ='' then null else kw1 end,
case when kw2 ='' then null else kw2 end,
case when kw3 ='' then null else kw3 end,
case when topic1 ='' then null else topic1 end,
case when topic2 ='' then null else topic2 end,
case when topic3 ='' then null else topic3 end,
case when appIdInstall ='' then null else appIdInstall end,
case when appIdAction ='' then null else appIdAction end,
case when ct ='' then null else ct end,
case when os ='' then null else os end,
case when carrier ='' then null else carrier end,
case when house='' then null else house end
from tmp.hqt_userFeature_2

---create table all
create table if not exists tmp.hqt_train_us_ad
as
select * from 
tmp.hqt_train a1
left join
tmp.hqt_userFeature a2
on 
a1.uid=a2.uid
left join 
tmp.hqt_adFeature a3
on
a1.aid=a3.aid

create table if not exists tmp.hqt_test1_us_ad
as
select * from 
tmp.hqt_test1 a1
left join
tmp.hqt_userFeature a2
on 
a1.uid=a2.uid
left join 
tmp.hqt_adFeature a3
on
a1.aid=a3.aid


---creat table train puls test1
create table if not exists tmp.hqt_data
as
select  * 
from
(
select aid,uid,label,age,gender,marriagestatus,education,consumptionability,lbs,
interest1,interest2,interest3,interest4,interest5,kw1,kw2,kw3,topic1,topic2,topic3,
appidinstall,appidaction,ct,os,carrier,house,advertiserid,campaignid,creativeid,creativesize,adcategoryid,productid,producttype
from tmp.hqt_train_us_ad 
UNION  ALL 
select aid,uid, -2 as label,age,gender,marriagestatus,education,consumptionability,lbs,
interest1,interest2,interest3,interest4,interest5,kw1,kw2,kw3,topic1,topic2,topic3,
appidinstall,appidaction,ct,os,carrier,house,advertiserid,campaignid,creativeid,creativesize,adcategoryid,productid,producttype
from tmp.hqt_test1_us_ad
) a


--把null改成-1




create table if not exists tmp.hqt_data_pnull
as
select aid,uid, 
case when label=-1 then 0 else label end as label,
case when age is null then '-1' else age end as age,
case when gender is null then '-1' else gender end as gender,
case when marriagestatus is null then '-1' else marriagestatus end as marriagestatus,
case when education is null then '-1' else education end as education,
case when consumptionability is null then '-1' else consumptionability end as consumptionability,
case when lbs is null then '-1' else lbs end as lbs ,
case when interest1 is null then '-1' else interest1 end as interest1,
case when interest2 is null then '-1' else interest2 end as interest2,
case when interest3 is null then '-1' else interest3 end as interest3,
case when interest4 is null then '-1' else interest4 end as interest4,
case when interest5 is null then '-1' else interest5 end as interest5,
case when kw1 is null then '-1' else kw1 end as kw1,
case when kw2 is null then '-1' else kw2 end as kw2,
case when kw3 is null then '-1' else kw3 end as kw3,
case when topic1 is null then '-1' else topic1 end as topic1,
case when topic2 is null then '-1' else topic2 end as topic2,
case when topic3 is null then '-1' else topic3 end as topic3,
case when appidinstall is null then '-1' else appidinstall end as appidinstall,
case when appidaction is null then '-1' else appidaction end as appidaction,
case when ct is null then '-1' else ct end as ct,
case when os is null then '-1' else os end as os,
case when carrier is null then '-1' else carrier end as carrier,
case when house is null then '-1' else house end as house,
advertiserid,campaignid,creativeid,creativesize,adcategoryid,productid,producttype
from tmp.hqt_data 


create table if not exists tmp.hqt_train_us_ad_pnull
as
select aid,uid,label,
case when age is null then '-1' else age end as age,
case when gender is null then '-1' else gender end as gender,
case when marriagestatus is null then '-1' else marriagestatus end as marriagestatus,
case when education is null then '-1' else education end as education,
case when consumptionability is null then '-1' else consumptionability end as consumptionability,
case when lbs is null then '-1' else lbs end as lbs ,
case when interest1 is null then '-1' else interest1 end as interest1,
case when interest2 is null then '-1' else interest2 end as interest2,
case when interest3 is null then '-1' else interest3 end as interest3,
case when interest4 is null then '-1' else interest4 end as interest4,
case when interest5 is null then '-1' else interest5 end as interest5,
case when kw1 is null then '-1' else kw1 end as kw1,
case when kw2 is null then '-1' else kw2 end as kw2,
case when kw3 is null then '-1' else kw3 end as kw3,
case when topic1 is null then '-1' else topic1 end as topic1,
case when topic2 is null then '-1' else topic2 end as topic2,
case when topic3 is null then '-1' else topic3 end as topic3,
case when appidinstall is null then '-1' else appidinstall end as appidinstall,
case when appidaction is null then '-1' else appidaction end as appidaction,
case when ct is null then '-1' else ct end as ct,
case when os is null then '-1' else os end as os,
case when carrier is null then '-1' else carrier end as carrier,
case when house is null then '-1' else house end as house,
advertiserid,campaignid,creativeid,creativesize,adcategoryid,productid,producttype
from tmp.hqt_train_us_ad 