import sys


class CSVFile:
    def __init__(self, file):
        self.cols = [col.strip() for col in next(file).split(',')]
        self.data = []
        for line in file:
            if len(line.strip()) > 0:
                cols = [col.strip() for col in line.split(',')]
                self.data.append(cols)
    
    def select(self, column_name, value):
        col_idx = self.cols.index(column_name)
        results = []
        for row in self.data:
            if row[col_idx] == value:
                results.append(row)
        return results


class Lineage(CSVFile):
    def __init__(self, file):
        return super().__init__(file)
    
    def max_frame(self):
        col_idx = self.cols.index('file')
        return max(self.data, key=lambda row: row[col_idx])[col_idx]

    def compare(self, other):
        my_max_frame = self.max_frame()
        other_max_frame = other.max_frame()
        if (my_max_frame != other_max_frame):
            print("Ended at wrong image frame!", file=sys.stderr)
            return False

        my_last_frame_data = self.select('file', my_max_frame)
        other_last_frame_data = other.select('file', my_max_frame)
        if abs(len(my_last_frame_data) - len(other_last_frame_data)) > 1:
            print("Difference in number of bacteria too big!", file=sys.stderr)
            return False

        my_points = [(float(row[2]), float(row[3])) for row in my_last_frame_data]
        other_points = [(float(row[2]), float(row[3])) for row in other_last_frame_data]
        for x,y in my_points:
            (x2,y2) = min(other_points, key=lambda pt: dist(x, y, *pt))
            if dist(x, y, x2, y2) > 25:
                print("Distance between pair of positions too big!", file=sys.stderr)
                return False

        return True


def dist(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


def main():
    try:
        with open(sys.argv[1]) as file:
            lineage1 = Lineage(file)
        with open(sys.argv[2]) as file:
            lineage2 = Lineage(file)
    except IndexError:
        print("Not enough arguments!", file=sys.stderr)
        sys.exit(1)
    except IOError:
        print("Failed to open files!", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if lineage1.compare(lineage2) else 1)


if __name__ == '__main__':
    main()