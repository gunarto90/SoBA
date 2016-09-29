import sys

class Venue:
    def __init__(self, _id, _lat, _lon):
        self.vid = _id               # Venue id
        self.lat = _lat
        self.lon = _lon 
        self.count = 0
        self.cluster = -1

    def set_count(self, count):
        self.count = count

    def increase_count(self):
        self.count += 1

    def set_cluster(self, cluster_id):
        self.cluster = cluster_id

    def __str__(self):
        return '{},{},{}'.format(self.vid, self.lat, self.lon)

class Checkin:
    def __init__(self, _uid, _vid, _lat, _lon, _time):
        self.uid = _uid
        self.vid = _vid
        self.lat = _lat
        self.lon = _lon
        self.time = _time

    def __str__(self):
        return '{},{},{},{},{}'.format(self.uid, self.time, self.lat, self.lon, self.vid)

    def __gt__(self, other):
        return self.time > other.time

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return self.u1 == other.u1 and self.u2 == other.u2

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

class Friend:
    def __init__(self, u1, u2):
        if u1 <= u2:
            self.u1 = u1
            self.u2 = u2
        else:
            self.u1 = u2
            self.u2 = u1

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return self.u1 == other.u1 and self.u2 == other.u2

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        return '{},{}'.format(self.u1, self.u2)

class User:
    def __init__(self, _id):
        self.uid = _id              # User id
        self.checkins = []
        self.friends = []
        self.dist = []
        self.earliest = sys.maxsize # Earliest checkin
        self.latest = 0             # Latest checkin

    def add_checkin(self, _vid, _lat, _lon, _time):
        self.checkins.append(Checkin(self.uid, _vid, _lat, _lon, _time))
        if _time < self.earliest:
            self.earliest = _time
        if _time > self.latest:
            self.latest = _time

    def add_friends(self, friend_list):
        for fid in friend_list:
            self.friends.append(fid)

    def add_distribution(self, venues, n_clusters):
        self.dist = []
        for i in range(0, n_clusters):
            self.dist.append(0)
        for c in self.checkins:
            vid = c.vid
            venue = venues.get(vid)
            if venue is None or venue.cluster == -1:
                continue
            self.dist[venue.cluster] += 1

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return self.uid == other.uid

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        return '{},{}'.format(self.id, len(self.checkins))