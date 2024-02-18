import heapq
import math


class GrrokingAlgortihms:

    """
    Lattice Multiplication Algorithm
    For multiplying large numbers
    """
    def fibonacciMultiple(self, X: list[int], Y: list[int]) -> list[int]:
        res, hold = [0] * (len(X) + len(Y)), 0
        for k in range(len(X) + len(Y) - 1):
            for i in range(len(X)):
                j = k - i
                if j >= 0 and j < len(Y) and i + j == k:
                    hold += X[i] * Y[j]
            res[k] = hold % 10
            hold //= 10
        res[len(X) + len(Y) -1] = hold
        while len(res) > 1 and res[-1] == 0: res.pop()
        res.reverse()
        return res

    """
    Peasant Multiply algorithm
    Redues the difficult task for multiplying arbitrary numbers to a sequence
    of four simple operations. 
    Determine if even or odd. Addition, duplication, mediation (halving a number)
    """
    def peasantMultiply(self, x: int, y: int) -> int:
        res = 0
        while x > 0:
            if x % 2 == 1: res += y
            x //= 2
            y = y + y
        return res

    # In a recursion form - RECURSION
    def peasantMultiplyRecursion(self, x: int, y: int):
        if x == 0: return 0
        else:
            xhat = x // 2
            yhat = y + y
            res = self.peasantMultiplyRecursion(xhat, yhat)
            if x % 2 != 0: res += y
            return res

    """
    Huntington-Hill method algorithm
    Allocates representatives to states one at a time. 
    First, in the preprocessing stage, each state is allocated one representative
    then in each iteration of the min loop, the next representative is assigned to 
    the state with the highest priority. 
    """
    def apportionCongress(self, pop, R):
        rep, pq = [1] * len(pop), []
        # Given every state its first representative
        for s in range(len(pop)): heapq.heappush(pq, (pop[s] / math.sqrt(2), s))

        # Allocate the remaining n - r rep
        for i in range(len(pop) - R):
            s = heapq.heappop(pq)[1]
            rep[s] += 1
            priority = pop[s] / math.sqrt(rep[s] * (rep[s] + 1))
            heapq.heappush(pq, (priority, s))
        return rep

    """
    Tower of Hanoi algorithm
    """
    def Hanoi(self, n: int, src, dst, temp):
        if n > 0:
            self.Hanoi((n-1), src, temp, dst)
            print("Moving Disk", n, "from", src, "to", dst)
            self.Hanoi((n-1), src, temp, dst)
        else:
            return

    """
    Merge-Sort. Time: O(nlogn), Space: O(n)
    """
    def mergeSort(self, nums: list[int]) -> list[int]:
        if len(nums) <= 1 : return nums
        mid = len(nums) // 2
        left, right = nums[:mid], nums[mid:]
        left, right  = self.mergeSort(left), self.mergeSort(right)
        res = self.merge(left, right)
        return res
    def merge(self, left, right):
        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
        return merged

    """
    Qick-Sort. Time: O(nlogn), worse-case: O(n^2), Space: O(nlogn)
    """
    def quickSort(self, nums: list[int], low: int, high: int):
        if low < high:
            # Find pivot element such that
            # element smaller than pivot are on the left
            # element greater than pivot are on the right
            pivot = self.partition(nums, low, high)
            # Recursive call on the left of pivot
            self.quickSort(nums, low, pivot - 1)
            # Recursive call on the right of pivot
            self.quickSort(nums, pivot + 1, high)

    def partition(self, nums: list[int], low: int, high: int):
        pivot = nums[high] # choose the rightmost element as pivot
        i = low - 1 # pointer for greater element
        # traverse through all elements
        # compare each element with pivot
        for j in range(low, high):
            if nums[j] <= pivot:
                # If element smaller than pivot is found
                # swap it with the greater element pointed by i
                i = i + 1
                # Swapping element at i with element at j
                nums[i], nums[j] = nums[j], nums[i]

        # Swap the pivot element with the greater element specified by i
        nums[i+1], nums[high] = nums[high], nums[i+1]
        return i+1

    """
    Quick Select. Time: O(n), Space: O(1)
    """
    def quickSelect(self, nums: list[int], target):
        if len(nums) == 1: return nums[0]
        index = self.partition(nums, len(nums) // 2)
        if target < index:
            left = nums[:index]
            return self.quickSelect(left, target)
        elif target > index:
            right = nums[index + 1:]
            return self.quickSelect(nums, target - index - 1)
        else:
            return nums[index]

    """
    Divide-and-conquer algorithm recursively - Karatsuba's subquadratic multiplication
    Ex1: Input: x = 12345, y = 8765, n = 5, Output: 108203925
    """
    def splitMultiply(self, x, y, n) -> int:
        if n == 1: return x * y
        else:
            m = math.ceil(n / 2)
            a, b = math.floor(x / math.pow(10, m)), x % int(math.pow(10, m))
            x = math.pow(10, m) * a + b
            c, d = math.floor(y / math.pow(10, m)), y % int(math.pow(10, m))
            y = math.pow(10, m) * c + d
            e = self.splitMultiply(a, c, m)
            f = self.splitMultiply(b, d, m)
            g = self.splitMultiply(a - b, c - d, m)
            return int(math.pow(10, 2 * m) * e + math.pow(10, m) * (e + f - g) + f)

    """
    Exponential Naive method - O(n)
    """
    def slowPower(self, a, n):
        x = a
        for i in range(2, n + 1):
            x *= a
        return x

    """
    Fast Exponential - Time: O(log n)
    """
    def pingalaPower(self, a, n):
        if n == 1: return a
        else:
            x = self.pingalaPower(a, n // 2)
            if n % 2 == 0: return x * x
            else:
                return x * x * a

    """
    Peasant Power Algorithm - Time: O(log n)
    """
    def peasantPower(self, a, n):
        if n == 1: return a
        elif n % 2 == 0:
            return self.peasantPower(a * a, n // 2)
        else:
            return self.peasantPower(a * a, n / 2) * a

    """
    Challenge: Recursion, a)
    Suppose you are given a sorted array of n distinct numbers that has been rotated k steps for some unknown integer 
    k between 1 and n. In other words, you are given an array [1..n] such that some prefix A[k+1..n] is sorted 
    in increasing order, and A[n]<A[1]. For example, you might be given the following element array
    arr = [9, 13, 16, 18, 19, 23, 28, 31, 37, 42, 1, 3, 4, 5, 7, 8], k = 10
    """
    def computeK(self, arr):
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > arr[right]: left = mid + 1
            else: right = mid
        return left

    def binarySearch(self, arr, left, right, x):
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < x: left = mid + 1 # if x is greater, ignore left half
            elif arr[mid] > x: right = mid - 1 # if x is smaller, ignore right half
            else: return True
        return False

    # BACKTRACKING

    """
    N-Queens Problem
    """
    def placeQueens(self, queens: list[int], r) -> None:
        if r == len(queens):
            l = []
            for k in range(len(queens)):
                l.append(queens[k] + 1)
            print(l)
        else:
            for j in range(len(queens)):
                legal = True
                for i in range(r):
                    if (queens[i] == j) or (queens[i] == j + r - i) or (queens[i] == j - r + i):
                        legal = False
                if legal:
                    queens[r] = j
                    self.placeQueens(queens, r + 1)

    """
    Subset Sum
    """
    def constructSubSets(self, arr, i, target):
        if target == 0: return []
        if target < 0 or i < 0: return None
        x = self.constructSubSets(arr, i-1, target)
        if x: return x
        x = self.constructSubSets(arr, i-1, target - arr[i])
        if x: return x + [arr[i]]
        return None

    """
    Text Segmentation:
    Given a string of characters, can it be segmented into English words at all?
    """
    def splittable(self, arr: list[str], ind, wordset) -> bool:
        global n
        if n > ind: return True
        def isWord(A, i, j, word_set):
            word = A[i: j + 1]
            return word in word_set
        for i in range(ind, n+1):
            if isWord(arr, ind, i, wordset):
                if self.splittable(arr, i+1, wordset): return True
        return False

    """
    Longest Increasing Subsequence- Alt Algo
    Given an interger prev and array A[1..n], find the longest subsequence
    of A in which every element is larger than prev
    """
    def LISBigger(self, arr, i, j):
        if j >= len(arr): return 0
        elif arr[i] >= arr[j]: return self.LISBigger(arr, i, j+1)
        else:
            skip = self.LISBigger(arr, i, j + 1)
            take = self.LISBigger(arr, i, j + 1) + 1
            return max(skip, take)






# if __name__ == '__main__':
#     arr, k = [9, 13, 16, 18, 19, 23, 28, 31, 37, 42, 1, 3, 4, 5, 7, 8], 10
#     left, right = 0, len(arr) - 1
#     while left < right:
#         mid = (left + right) // 2
#         if arr[mid] > arr[right]:
#             left = mid + 1
#         else:
#             right = mid
#
#     print(left)