
import argparse
import csv
import os
from textgrid import *

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Convert a VOT tier fo a TextGrid to a CSV file. The CSV file will '
                                                 'contain the filename, the duration of the mark, and the mark name.')
    parser.add_argument('textgrid_filename', help="name of an input TextGrid file")
    parser.add_argument('csv_filename', help="name of an output CSV file.")
    parser.add_argument('tier', help='the tier name of the TextGrid that should be converted to CSV.')
    args = parser.parse_args()


    out_file = open(args.csv_filename, 'wb')
    csv_file = csv.writer(out_file)
    csv_file.writerow(['textgrid_file','time','vot','mark'])

    # read TextGrid
    textgrid = TextGrid()
    textgrid.read(args.textgrid_filename)

    # extract tier names
    tier_names = textgrid.tierNames()

    basename = os.path.splitext(os.path.basename(args.textgrid_filename))[0]

    # check if the VOT tier is one of the tiers in the TextGrid
    if args.tier in tier_names:
        tier_index = tier_names.index(args.tier)
        # run over all intervals in the tier
        for interval in textgrid[tier_index]:
            if re.search(r'\S', interval.mark()):
                intervals = list()
                intervals.append(basename)
                intervals.append("{:.3f}".format(interval.xmin()))
                intervals.append("{:.3f}".format(interval.xmax()-interval.xmin()))
                intervals.append(interval.mark())
                csv_file.writerow(intervals)
                #print intervals
    # close CSV file
    out_file.close()

