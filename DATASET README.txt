#############################################
############### Description #################
#############################################

user.csv
uid, row_uid, #checkins, row_checkins, #friends, row_friends

user_weekend.csv
uid, #checkins

friend.csv
user_1, user_2

checkin.csv
user_id, timestamp, latitude, longitude, venue_id

venue.csv
vid, latitude, longitude

#############################################
################## All data #################
#############################################

@#Checkins
Gowalla		: 6,442,727
Brightkite	: 4,491,074

@#Venues
Gowalla		: 1,280,809
Brightkite	: 772,703

@#Users
Gowalla		: 107,068 (107,092 in dataset)
Brightkite	: 50,686 (58,228 in dataset)

@#Active Users (#Checkins > 100)
Gowalla		: 14,796
Brightkite	: 8,173

@#Friendship
Gowalla		: 1,900,640	(Only 913,532 ties among who have checkins)
Brightkite	: 428,156	(Only 388,180 ties among who have checkins)

@Diversity data (maximum unique visitor in a venue)

Gowalla		: 117
Brightkite	: 400

#############################################
############### Weekends data ###############
############################################# 

@#Checkins
Gowalla		: 2,013,870 (out of all 6,442,727)	31.26%
Brightkite	: 1,334,403 (out of all 4,491,074)	29.71%

@#Venues
Gowalla		: 78,237 (out of all 1,280,809)		6.10%
Brightkite	: 28,158 (out of all 772,703)		3.64%

@#Users
Gowalla		: 92,988
Brightkite	: 35,390

@#Active Users (#Checkins > 100)
Gowalla		: 3,237
Brightkite	: 3,153

@#Friendship
Gowalla		: 804,957
Brightkite	: 305,670