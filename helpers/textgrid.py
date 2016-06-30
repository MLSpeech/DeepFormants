# This file is a slightly modified version of the textgrid.py module
# (https://github.com/kylebgorman/textgrid/), which was released under the following license:
# (see https://github.com/kylebgorman/textgrid/blob/master/LICENSE)
#
# Copyright (c) 2011-2013 Kyle Gorman, Max Bane, Morgan Sonderegger
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.



import logging
import re


class mlf:
    """
    read in a HTK .mlf file. iterating over it gives you a list of 
    TextGrids
    """

    def __init__(self, file):
        self.__items = []
        self.__n = 0
        text = open(file, 'r')
        text.readline() # get rid of header
        while 1: # loop over text
            name = text.readline()[1:-1]
            if name:
                grid = TextGrid()
                phon = IntervalTier('phones')
                word = IntervalTier('words')
                wmrk = ''
                wsrt = 0.
                wend = 0.
                while 1: # loop over the lines in each grid
                    line = text.readline().rstrip().split()
                    if len(line) == 4: # word on this baby
                        pmin = float(line[0]) / 10e6
                        pmax = float(line[1]) / 10e6
                        phon.append(Interval(pmin, pmax, line[2]))
                        if wmrk:
                            word.append(Interval(wsrt, wend, wmrk))
                        wmrk = line[3]
                        wsrt = pmin
                        wend = pmax
                    elif len(line) == 3: # just phone
                        pmin = float(line[0]) / 10e6
                        pmax = float(line[1]) / 10e6
                        phon.append(Interval(pmin, pmax, line[2]))
                        wend = pmax 
                    else: # it's a period
                        word.append(Interval(wsrt, wend, wmrk))
                        self.__items.append(grid)
                        break
                grid.append(phon)
                grid.append(word)
                self.__n += 1
            else:
                text.close()
                break

    def __iter__(self):
        return iter(self.__items)

    def __len__(self):
        return self.__n

    def __str__(self):
        return '<MLF instance with %d TextGrids>' % self.__n

class TextGrid:
    """ represents Praat TextGrids as list of different types of tiers """

    def __init__(self, name = None): 
        self.__tiers = []
        self.__n = 0
        self.__xmin = None
        self.__xmax = None
        self.__name = name # this is just for the MLF case

    def __str__(self):
        return '<TextGrid with %d tiers>' % self.__n

    def __iter__(self):
        return iter(self.__tiers)

    def __len__(self):
        return self.__n

    def __getitem__(self, i):
        """ return the (i-1)th tier """
        return self.__tiers[i] 

    # Morgan Sonderegger
    def tierNames(self, case=None):
        names = [t.name() for t in self.__tiers]
        if(case=="lower"):
            names = [n.lower() for n in names]
        return names

    def xmin(self):
        return self.__xmin

    def xmax(self):
        return self.__xmax

    def append(self, tier):
        self.__tiers.append(tier)
        ## JosephKeshet
        if self.__xmin is None:
            self.__xmin = tier.xmin()
        else:
            self.__xmin = min(tier.xmin(), self.__xmin)
        ## JosephKeshet
        self.__xmax = max(tier.xmax(), self.__xmax)
        ## JosephKeshet / MS
        if self.__xmax is None:
            self.__xmax = tier.xmax()
        else:
            self.__xmax = max(tier.xmax(), self.__xmax)
        self.__n += 1

    def read(self, file):
        """ read TextGrid from Praat .TextGrid file """
        text = open(file, 'r')
        text.readline() # header crap
        text.readline()
        text.readline()
        self.__xmin = float(text.readline().rstrip().split()[2])
        self.__xmax = float(text.readline().rstrip().split()[2])
        text.readline()
        m = int(text.readline().rstrip().split()[2]) # will be self.__n soon
        text.readline()
        for i in range(m): # loop over grids
            text.readline()
            if text.readline().rstrip().split()[2] == '"IntervalTier"':
                # inam = text.readline().rstrip().split()[2][1:-1]
                inam = text.readline().split('=')[1].strip().strip('"') # Joseph Keshet: handle space in the tier name
                imin = float(text.readline().rstrip().split()[2])
                imax = float(text.readline().rstrip().split()[2])
                itie = IntervalTier(inam, imin, imax) # redundant FIXME
                n = int(text.readline().rstrip().split()[3])
                for j in range(n):
                    try:
                        text.readline().rstrip().split() # header junk
                        jmin = float(text.readline().rstrip().split()[2])
                        jmax = float(text.readline().rstrip().split()[2])
                        # Morgan Sonderegger changed, to account for intervals where label
                        # begins with spacing
                        #jmrk = text.readline().rstrip().split()[2][1:-1]
                        #jmrk = text.readline().split('=')[1].strip().strip('"') # Joseph Keshet: handle space in the
                        # tier
                        # name
                        jmrk = getMark(text)
                        #
                        itie.append(Interval(jmin, jmax, jmrk))
                    except:
                        logging.error("Unable to parse TextGrid %s." % text.name)

                self.append(itie) 
            else: # pointTier
                # inam = text.readline().rstrip().split()[2][1:-1]
                inam = text.readline().split('=')[1].strip().strip('"') # Joseph Keshet: handle space in the tier name
                imin = float(text.readline().rstrip().split()[2])
                imax = float(text.readline().rstrip().split()[2])
                itie = PointTier(inam, imin, imax) # redundant FIXME
                n = int(text.readline().rstrip().split()[3])
                for j in range(n):
                    text.readline().rstrip() # header junk
                    jtim = float( text.readline().rstrip().split()[2])
                    jmrk = text.readline().rstrip().split()[2][1:-1]
                    itie.append(Point(jtim, jmrk))
                self.append(itie)
        text.close()

    def write(self, text):
        """ write it into a text file that Praat can read """
        text = open(text, 'w')
        text.write('File type = "ooTextFile"\n')
        text.write('Object class = "TextGrid"\n\n')
        text.write('xmin = %f\n' % self.__xmin)
        text.write('xmax = %f\n' % self.__xmax)
        text.write('tiers? <exists>\n')
        text.write('size = %d\n' % self.__n)
        text.write('item []:\n')
        for (tier, n) in zip(self.__tiers, range(1, self.__n + 1)):
            text.write('\titem [%d]:\n' % n)
            if tier.__class__ == IntervalTier: 
                text.write('\t\tclass = "IntervalTier"\n')
                text.write('\t\tname = "%s"\n' % tier.name())
                text.write('\t\txmin = %f\n' % tier.xmin())
                text.write('\t\txmax = %f\n' % tier.xmax())
                text.write('\t\tintervals: size = %d\n' % len(tier))
                for (interval, o) in zip(tier, range(1, len(tier) + 1)): 
                    text.write('\t\t\tintervals [%d]:\n' % o)
                    text.write('\t\t\t\txmin = %f\n' % interval.xmin())
                    text.write('\t\t\t\txmax = %f\n' % interval.xmax())
                    text.write('\t\t\t\ttext = "%s"\n' % interval.mark())
            else: # PointTier
                text.write('\t\tclass = "TextTier"\n')
                text.write('\t\tname = "%s"\n' % tier.name())
                text.write('\t\txmin = %f\n' % tier.xmin())
                text.write('\t\txmax = %f\n' % tier.xmax())
                text.write('\t\tpoints: size = %d\n' % len(tier))
                for (point, o) in zip(tier, range(1, len(tier) + 1)):
                    text.write('\t\t\tpoints [%d]:\n' % o)
                    text.write('\t\t\t\ttime = %f\n' % point.time())
                    text.write('\t\t\t\tmark = "%s"\n' % point.mark())
        text.close()

class IntervalTier:
    """ represents IntervalTier as a list plus some features: min/max time, 
    size, and tier name """

    def __init__(self, name = None, xmin = None, xmax = None):
        self.__n = 0
        self.__name = name
        self.__xmin = xmin
        self.__xmax = xmax
        self.__intervals = []

    def __str__(self):
        return '<IntervalTier "%s" with %d points>' % (self.__name, self.__n)

    def __iter__(self):
        return iter(self.__intervals)

    def __len__(self):
        return self.__n

    def __getitem__(self, i):
        """ return the (i-1)th interval """
        return self.__intervals[i]

    def xmin(self):
        return self.__xmin

    def xmax(self):
        return self.__xmax

    def name(self):
        return self.__name

    def append(self, interval):
        self.__intervals.append(interval)
        self.__xmax = interval.xmax()
        self.__n += 1

    # Morgan Sonderegger added
    def remove(self, interval):
        logging.debug("removing %d" % interval.xmin())
        self.__intervals.remove(interval)
        self.__n -= 1

    def read(self, file):
        text = open(file, 'r')
        text.readline() # header junk 
        text.readline()
        text.readline()
        self.__xmin = float(text.readline().rstrip().split()[2])
        self.__xmax = float(text.readline().rstrip().split()[2])
        self.__n = int(text.readline().rstrip().split()[3])
        for i in range(self.__n):
            text.readline().rstrip() # header
            imin = float(text.readline().rstrip().split()[2]) 
            imax = float(text.readline().rstrip().split()[2])
            # imrk = text.readline().rstrip().split()[2].replace('"', '') # txt
            imrk = text.readline().split('=')[1].strip().strip('"') # Joseph Keshet: handle space in the mark
            self.__intervals.append(Interval(imin, imax, imrk))
        text.close()

    def write(self, file):
        text = open(file, 'w')
        text.write('File type = "ooTextFile"\n')
        text.write('Object class = "IntervalTier"\n\n')
        text.write('xmin = %f\n' % self.__xmin)
        text.write('xmax = %f\n' % self.__xmax)
        text.write('intervals: size = %d\n' % self.__n)
        for (interval, n) in zip(self.__intervals, range(1, self.__n + 1)):
            text.write('intervals [%d]:\n' % n)
            text.write('\txmin = %f\n' % interval.xmin())
            text.write('\txmax = %f\n' % interval.xmax())
            text.write('\ttext = "%s"\n' % interval.mark())
        text.close()

class PointTier:
    """ represents PointTier (also called TextTier for some reason) as a list 
    plus some features: min/max time, size, and tier name """

    def __init__(self, name = None, xmin = None, xmax = None):
        self.__n = 0
        self.__name = name
        self.__xmin = xmin
        self.__xmax = xmax
        self.__points = []

    def __str__(self):
        return '<PointTier "%s" with %d points>' % (self.__name, self.__n)

    def __iter__(self):
        return iter(self.__points)
    
    def __len__(self):
        return self.__n
    
    def __getitem__(self, i):
        """ return the (i-1)th tier """
        return self.__points[i]

    def name(self):
        return self.__name

    def xmin(self):
        return self.__xmin

    def xmax(self): 
        return self.__xmax

    def append(self, point):
        self.__points.append(point)
        ## MS: points don't have xmax, right?
        # self.__xmax = point.xmax()
        if self.__xmax is None:
            self.__xmax = point.time()
        else:
            self.__max = max(point.time(), self.__xmax)
        ## MS: do we then need to do this for xmin as well?
        self.__n += 1

    def read(self, file):
        text = open(file, 'r')
        text.readline() # header junk 
        text.readline()
        text.readline()
        self.__xmin = float(text.readline().rstrip().split()[2])
        self.__xmax = float(text.readline().rstrip().split()[2])
        self.__n = int(text.readline().rstrip().split()[3])
        for i in range(self.__n):
            text.readline().rstrip() # header
            itim = float(text.readline().rstrip().split()[2])
            imrk = text.readline().rstrip().split()[2].replace('"', '') # txt
            self.__points.append(Point(imrk, itim))
        text.close()

    def write(self, file):
        text = open(file, 'w')
        text.write('File type = "ooTextFile"\n')
        text.write('Object class = "TextTier"\n\n')
        text.write('xmin = %f\n' % self.__xmin)
        text.write('xmax = %f\n' % self.__xmax)
        text.write('points: size = %d\n' % self.__n)
        for (point, n) in zip(self.__points, range(1, self.__n + 1)):
            text.write('points [%d]:\n' % n)
            text.write('\ttime = %f\n' % point.time())
            text.write('\tmark = "%s"\n' % point.mark())
        text.close()

class Interval:
    """ represent an Interval """
    def __init__(self, xmin, xmax, mark):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__mark = mark
    
    def __str__(self):
        return '<Interval "%s" %f:%f>' % (self.__mark, self.__xmin, self.__xmax)

    def xmin(self):
        return self.__xmin

    def xmax(self):
        return self.__xmax

    # Morgan Sonderegger added
    def bounds(self):
        return (self.__xmin, self.__xmax)
    
    def mark(self):
        return self.__mark

class Point:
    """ represent a Point """
    def __init__(self, time, mark):
        self.__time = time
        self.__mark = mark
    
    def __str__(self):
        return '<Point "%s" at %f>' % (self.__mark, self.__time)

    def time(self):
        return self.__time

    def mark(self):
        return self.__mark

# Morgan Sonderegger added: account for intervals with writing beginning with whitespace
#def correctLine(line):
def getMark(text):
    line = text.readline().rstrip()
    a = re.search('(\S+) (=) (".*")', line)
    assert(a)
    assert(len(a.groups())==3)
    return a.groups()[2][1:-1]
    
