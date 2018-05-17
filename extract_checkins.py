from general_utilities import *
from base import *
from classes import *

checkins_filename = 'p{}_k{}_10.checkin'

# Main function
if __name__ == '__main__':
    ### Global parameter for the experiments
    ps = []     ### Active project: 0 Gowalla, 1 Brightkite
    ks = []     ### Mode for top k users: 0 Weekend, -1 All users

    ### project to be included
    ps.append(0)
    ps.append(1)
    ### mode to be included
    ks.append(0)
    ks.append(-1)

    debug("--- Extract checkins started ---")
    for p in ps:
        for k in ks:
            debug('p:{}, k:{}'.format(p, k))
            ### Initialize variables
            dataset, base_folder, working_folder, weekend_folder = init_folder(p, k)
            dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
            # ### Initialize dataset
            users, friends, venues = init(p, k)
            # ### Sorting users' checkins based on their timestamp, ascending ordering
            uids = sort_user_checkins(users)
            if k == -1:
                folder = base_folder
            elif k == 0:
                folder = weekend_folder

            with open(working_folder + '/' + checkins_filename.format(p, k), 'w') as fw:
                mid = 1
                fw.write('mid,uid,locid\n')
                for uid, user in users.items():
                    for c in user.checkins:
                        fw.write('{},{},{}\n'.format(mid, c.uid, c.vid))
                        mid += 1