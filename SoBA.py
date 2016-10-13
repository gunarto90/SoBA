from general_utilities import *
from base import *
from classes import *

co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
co_location_filename = 'co_location_p{}_k{}_t{}_d{}.csv'
evaluation_filename = 'evaluation_p{}_k{}_t{}_d{}.csv'

def write_evaluation(summaries, p, k, t, d):
    texts = []
    texts.append('uid1,uid2,frequency,diversity,duration,stability,link')
    for friend, evaluation in summaries.items():
        texts.append(str(evaluation))
    filename = working_folder + evaluation_filename.format(p, k, t, d)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)
    debug('Finished writing evaluation to {}'.format(filename), out_file=False)

def extraction(p, k, t, d, working_folder):
    stat_f = {}     # frequency
    stat_d = {}     # diversity
    stat_ld = {}    # co-occurrence between user in each venue (distinct)
    stat_lv = {}    # co-occurrence between user in each venue
    stat_l = {}     # co-occurrence between user in each venue (dictionary)
    stat_t = {}     # time diff on co-occurrence
    stat_td = {}    # duration
    stat_ts = {}    # stability
    u_xy = {}       # u_xy = Average meeting time = delta_xy / |Theta_xy|
    week_num = {}   # Save a list of week number of co-occurrence between users
    ### Extract data from file
    fname = co_raw_filename.format(p, k, t, d)
    debug(fname)
    weeks = {}

    try:
        with open(working_folder + fname, 'r') as fr:
            for line in fr:
                split = line.strip().split(',')
                # user1,user2,vid,t_diff,frequency,time1,time2,t_avg
                if line.strip() == 'user1,user2,vid,t_diff,frequency,time1,time2,t_avg':
                    continue
                u1 = split[0]
                u2 = split[1]
                vid = split[2]
                t_diff = int(split[3])
                t_avg = float(split[len(split)-1])
                friend = Friend(u1, u2)
                ### Extract frequency
                found = stat_f.get(friend)
                if found is None:
                    found = 0
                stat_f[friend] = found + 1
                ### Extract diversity
                found = stat_l.get(friend)
                if found is None:
                    found = {}
                found_vid = found.get(vid)
                if found_vid is None:
                    found_vid = 0
                found_vid += 1
                found[vid] = found_vid
                stat_l[friend] = found
                ### Extract duration
                found = stat_t.get(friend)
                if found is None:
                    found = []
                found.append(t_diff)
                stat_t[friend] = found
                ### Extract co-occurrence on a venue
                found = stat_lv.get(friend)
                if found is None:
                    found = []
                found.append(vid)
                stat_lv[friend] = found
                if vid not in stat_lv:
                    stat_ld[friend] = found
                found = week_num.get(friend)
                if found is None:
                    found = []
                tanggal = date.fromtimestamp(t_avg)
                tahun = tanggal.year - 2009
                week = tanggal.isocalendar()[1] + (tahun * 53)
                found.append(week)
                week_num[friend] = found
        #         count = weeks.get(tanggal)
        #         if count is None:
        #             count = 0
        #         count += 1
        #         weeks[tanggal] = count
        # for w in sorted(weeks, key=weeks.get, reverse=True):
        #     debug('{}\t{}'.format(w, weeks[w]), clean=True)
        ### Extract diversity
        for friend, dictionary in stat_l.items():
            found = stat_d.get(friend)
            if found is None:
                found = []            
            for vid, x in dictionary.items():
                found.append(x)
            stat_d[friend] = found
        ### Extract duration
        duration = 0
        max_duration = 0
        for friend, data in stat_t.items():
            duration = max(data) - min(data)
            if duration > max_duration:
                max_duration = duration
            stat_td[friend] = duration
        for friend, data in stat_td.items():
            stat_td[friend] = float(stat_td[friend]) / float(max_duration)
            Theta_xy = stat_f.get(friend)
            if Theta_xy is None:
                debug('Error: Co-occurrence not found between {}'.format(friend))
                continue
            u_xy[friend] = float(stat_td.get(friend)) / float(Theta_xy)
        ### Extract stability
        sigma_z = {}    # Sum of t_xy^z-u_xy
        rho = {}        # standard deviation of co-occurrence time difference with average
        for friend, data in stat_t.items():
            for t in data:
                found = sigma_z.get(friend)
                if found is None:
                    found = 0
                mean = u_xy.get(friend)
                if mean is None:
                    debug('Error: mean u_xy not found between {}'.format(friend))
                    continue
                if mean == 0:
                    # debug('Error: mean u_xy is 0 between {}'.format(friend))
                    continue
                found += pow( (float(t) / float(max_duration)) - mean, 2)
                sigma_z[friend] = found
            ### Eliminate if frequency is only 1
            found = stat_f.get(friend)
            if found is not None and found <= 1:
                continue
            if sigma_z.get(friend) is None:
                # debug('Sigma z between {} is not found'.format(friend))
                continue
            if stat_ld.get(friend) is None:
                # debug('Distinct location count between {} is not found'.format(friend))
                continue
            rho[friend] = sqrt(sigma_z[friend] / len(stat_ld[friend]))
            if u_xy.get(friend) is None:
                continue
            stat_ts[friend] = exp(-1*(u_xy[friend]+rho[friend]))
            # debug(stat_ts[friend], clean=True)

        # for friend, weeks in week_num.items():
        #     debug('{}:{}'.format(friend, weeks))

        ### normalization of each weight
        # max_f = max(stat_f.values())
        # for friend, frequency in stat_f.items():
        #     stat_f[friend] = float(frequency) / max_f

        ### Debug
        ### Frequency
        # for friend, frequency in stat_f.items():
        #     debug('{},{}'.format(friend, frequency), clean=True)
        ### Entropy (Diversity)
        # for friend, data in stat_d.items():
        #     debug('{},{}'.format(friend, entropy(data)), clean=True)
        ### Duration
        # for friend, duration in stat_td.items():
        #     debug('{}\t{}'.format(friend, duration))
        ### Stability
        # for friend, weight in stat_ts.items():
        #     debug('{},{}'.format(friend, weight), clean=True)

        debug('Finished extracting co-occurrence features', out_file=False)

        return stat_f, stat_d, stat_td, stat_ts
    except:
        debug('{} is not found'.format(working_folder + fname))
        return None, None, None, None

def evaluation(friends, stat_f, stat_d, stat_td, stat_ts, p, k, t, d):
    summaries = {}

    if stat_f is None or stat_d is None or stat_td is None or stat_ts is None:
        return

    ### Normalization

    ### Frequency
    max_val = max(stat_f.values())
    for friend, data in stat_f.items():
        stat_f[friend] = float(data) / max_val
    for friend, data in stat_f.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.frequency = data
        summaries[friend] = found

    ### Diversity
    max_val = 0
    for friend, data in stat_d.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.diversity = entropy(data)
        if found.diversity > max_val:
            max_val = found.diversity
        summaries[friend] = found
    for friend, evaluation in summaries.items():
        evaluation.diversity = float(evaluation.diversity) / max_val

    ### Duration
    max_val = max(stat_td.values())
    for friend, data in stat_td.items():
        stat_td[friend] = float(data) / max_val
    for friend, data in stat_td.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.duration = data
        summaries[friend] = found

    ### Stability
    max_val = max(stat_ts.values())
    for friend, data in stat_ts.items():
        stat_ts[friend] = float(data) / max_val
    for friend, data in stat_ts.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.stability = data
        summaries[friend] = found

    ### Friendship
    for uid, arr in friends.items():
        for fid in arr:
            friend = Friend(uid, fid)
            found = summaries.get(friend)
            if found is None:
                continue
            found.link = 1
            summaries[friend] = found
            # debug(found, clean=True)
    # for friend, evaluation in summaries.items():
    #     debug(evaluation, clean=True)
    write_evaluation(summaries, p, k, t, d)

# Main function
if __name__ == '__main__':
    ### Global parameter for the experiments
    ps = []     ### Active project: 0 Gowalla, 1 Brightkite
    ks = []     ### Mode for top k users: 0 Weekend, -1 All users
    ts = []     ### Time threshold
    ds = []     ### Distance threshold
    ### project to be included
    ps.append(0)
    ps.append(1)
    ### mode to be included
    ks.append(0)
    ks.append(-1)
    ### time threshold to be included
    HOUR  = 3600
    DAY   = 24 * HOUR
    WEEK  = 7 * DAY
    MONTH = 30 * DAY
    ts.append(int(0.5 * HOUR))
    ts.append(1 * HOUR)
    ts.append(int(1.5 * HOUR))
    ts.append(2 * HOUR)
    # ts.append(1 * DAY)
    # ts.append(2 * DAY)
    # ts.append(3 * DAY)
    # ts.append(1 * WEEK)
    # ts.append(2 * WEEK)
    # ts.append(1 * MONTH)
    # ts.append(2 * MONTH)
    ### distance threshold to be included
    ds.append(0)
    ds.append(250)
    ds.append(500)
    ds.append(750)
    # ds.append(1000)
    debug("--- Social inference started ---")
    for t in ts:
        for d in ds:
            for k in ks:
                for p in ps:
                    debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d))
                    ### Initialize variables
                    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                    dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
                    # ### Initialize dataset
                    users, friends, venues = init(p, k)
                    # ### Sorting users' checkins based on their timestamp, ascending ordering
                    uids = sort_user_checkins(users)
                    ### Feature extraction
                    stat_f, stat_d, stat_td, stat_ts = extraction(p, k, t, d, working_folder)
                    evaluation(friends, stat_f, stat_d, stat_td, stat_ts, p, k, t, d)
                    ### testing extracted csv
                    # testing(p, k, t, d, working_folder)
    debug("--- Social inference finished ---")


