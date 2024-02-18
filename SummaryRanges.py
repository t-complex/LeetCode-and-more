class SummaryRanges:

    """
    352. Data Stream as Disjoint Intervals

    Given a data stream input of non-negative integers a1, a2, ..., an,
    summarize the numbers seen so far as a list of disjoint intervals.

    Implement the SummaryRanges class:

    SummaryRanges() Initializes the object with an empty stream.
    void addNum(int value) Adds the integer value to the stream.
    int[][] getIntervals() Returns a summary of the integers in the stream currently
    as a list of disjoint intervals [starti, endi]. The answer should be sorted by starti.

    Example 1:

    Input
    ["SummaryRanges", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals"]
    [[], [1], [], [3], [], [7], [], [2], [], [6], []]
    Output
    [null, null, [[1, 1]], null, [[1, 1], [3, 3]], null, [[1, 1], [3, 3], [7, 7]], null, [[1, 3], [7, 7]], null, [[1, 3], [6, 7]]]

    Explanation
    SummaryRanges summaryRanges = new SummaryRanges();
    summaryRanges.addNum(1);      // arr = [1]
    summaryRanges.getIntervals(); // return [[1, 1]]
    summaryRanges.addNum(3);      // arr = [1, 3]
    summaryRanges.getIntervals(); // return [[1, 1], [3, 3]]
    summaryRanges.addNum(7);      // arr = [1, 3, 7]
    summaryRanges.getIntervals(); // return [[1, 1], [3, 3], [7, 7]]
    summaryRanges.addNum(2);      // arr = [1, 2, 3, 7]
    summaryRanges.getIntervals(); // return [[1, 3], [7, 7]]
    summaryRanges.addNum(6);      // arr = [1, 2, 3, 6, 7]
    summaryRanges.getIntervals(); // return [[1, 3], [6, 7]]

    """


    def __init__(self): return None

    def addNum(self, value: int) -> None: return None

    def getIntervals(self) -> list[list[int]]: return None

    # def __init__(self):
    #     self.intervals = []
    # def _merge_intervals(self, interval):
    #     if len(self.intervals) == 0:
    #         self.intervals.append(interval)
    #         return
    #     intervals, i = [], 0
    #     while i < len(self.intervals):
    #         if interval[0] <= self.intervals[i][1] + 1: break
    #         intervals.append(self.intervals[i])
    #         i += 1
    #     merged = interval
    #     while i < len(self.intervals):
    #         if self.intervals[i][0] <= merged[1] + 1:
    #             merged = [min(self.intervals[i][0], merged[0]), max(self.intervals[i][1], merged[1])]
    #         else: break
    #         i += 1
    #     intervals.append(merged)
    #     while i < len(self.intervals):
    #         intervals.append(self.intervals[i])
    #         i += 1
    #     self.intervals = intervals
    # def addNum(self, value: int) -> None:
    #     self._merge_intervals([value, value])
    # def getIntervals(self) -> list[list[int]]:
    #     return self.intervals