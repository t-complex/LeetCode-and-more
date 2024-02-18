import bisect
import itertools
import collections
import functools
import heapq
import math
from collections import defaultdict, Counter, deque
from UnionFind import UnionFind
from GraphNode import Node
from Trie import TrieNode, Trie
from sortedcontainers import SortedList
from TreeNode import TreeNode
from ListNode import ListNode
from ListNodeWithRandomPointer import ListNodeWithRandomPointer


class LeetCodeProblems:

    #################### Arrays & Hashing #################### 9/9 - 5

    def contains_duplicate(self, nums: list) -> bool:
        set_num = set()
        for num in nums:
            if num in set_num: return True
            set_num.add(num)
        return False

    def is_anagram(self, s: str, t: str) -> bool:
        # if len(s) != len(t): return False
        # countS, countT = {}, {}
        # for i in range(len(s)):
        #     countS[s[i]] = 1 + countS.get(s[i], 0)
        #     countT[t[i]] = 1 + countT.get(t[i], 0)
        # return countS == countT
        map = {}  # <- a bit faster
        for c in s: map[c] = 1 + map.get(c, 0)
        for c in t:
            if c not in map: return False
            map[c] -= 1
            if map[c] == 0: del map[c]
        return len(map) == 0

    def two_sum(self, nums: list, target: int) -> list:
        mapp = dict()
        for i, num in enumerate(nums):
            x = target - num
            if x in mapp: return [mapp[num], i]
            mapp[num] = i
        return None

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        mapp = collections.defaultdict(list)
        for s in strs: mapp["".join(sorted(s))].append(s)
        return list(mapp.values())

    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        count = Counter(nums).most_common(k)  # --> this is O(klogn)
        return [num[0] for num in count]
        # count = {}
        # freq = [[] for i in range(len(nums) + 1)]
        # for n in nums: count[n] = 1 + count.get(n, 0)
        # for n, c in count.items(): freq[c].append(n)
        # res = []
        # for i in range(len(freq) - 1, 0, -1):
        #     for n in freq[i]:
        #         res.append(n)
        #         if len(res) == k: return res

    def productExceptSelf(self, nums: list[int]) -> list[int]:
        result, rightP = [1] * (len(nums)), 1
        for i in range(1, len(nums)): result[i] = result[i - 1] * nums[i - 1]
        for i in range(len(nums) - 2, -1, -1):
            rightP = rightP * nums[i + 1]
            result[i] = rightP * result[i]
        return result

    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows, cols, squa = defaultdict(set), defaultdict(set), defaultdict(set)  # key=(r//3,c//3)
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".": continue
                if (board[r][c] in rows[r] or board[r][c] in cols[c] or
                        board[r][c] in squa[(r // 3, c // 3)]): return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squa[(r // 3, c // 3)].add(board[r][c])
        return True

    def encode(self, strs):
        return ''.join("#" + str(len(s)) + s for s in strs)

    def decode(self, strs):
        # res, i = [], 0
        # while i < len(strs):
        #     j = i
        #     while strs[j] != "#": j += 1
        #     length = int(strs[i:j])
        #     res.append(strs[j + 1 : j + 1 + length])
        #     i = j + 1 + length
        # return res
        res, i = [], 0
        while i < len(strs):
            if strs[i] == '#':
                length = int(strs[i + 1])
                res.append(strs[i + 2: i + 2 + length])
                i = i + 2 + length
        return res

    def longestConsecutive(self, nums: list[int]) -> int:
        numSet, long = set(nums), 0
        for n in numSet:
            if (n - 1) not in numSet:
                l = 1
                while (n + l) in numSet: l += 1
                long = max(long, l)
        return long

    # new Picks from 300 list

    def replaceElements(self, arr: list[int]) -> list[int]:
        # rightMax = -1
        # for i in range(len(arr) -1, -1, -1):
        #     newMax = max(rightMax, arr[i])
        #     arr[i] = rightMax
        #     rightMax = newMax
        # return arr
        temp = arr[len(arr) - 1]
        arr[len(arr) - 1] = -1
        for i in range(len(arr) - 2, -1, -1):
            cur = arr[i]
            arr[i] = temp
            temp = max(temp, cur)
        return arr

    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]: i += 1
            j += 1
        return i == len(s)

    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])  # One-liner shortcut

    def longestCommonPrefix(self, strs: list[str]) -> str:
        res, sort = "", sorted(strs)
        first, last = sort[0], sort[-1]
        for i in range(min(len(first), len(last))):
            if first[i] != last[i]: return res
            res += first[i]
        return res

    def generate(self, numRows: int) -> list[list[int]]:
        tri = []
        for i in range(numRows):
            row = []
            for k in range(i + 1): row.append(math.comb(i, k))
            tri.append(row)
        return tri

    def fullyJustify(self, words: list[int], maxWidth: int) -> list[str]:
        res, line, width = [], [], 0
        for w in words:
            if width + len(w) + len(line) > maxWidth:
                for i in range(maxWidth - width): line[i % (len(line) - 1 or 1)] += ' '
                res, line, width = res + [''.join(line)], [], 0
            line += [w]
            width += len(w)
        return res + [' '.join(line).ljust(maxWidth)]

    #################### Two Pointer #################### 5/5 (1 HARD) - 5

    def isPalindrome(self, s: str) -> bool:
        temp = ("".join(i for i in s if i.isalnum())).lower()
        return temp == temp[::-1]

    def two_sum_input_array(self, numbers: list[int], target: int) -> list[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            curSum = numbers[l] + numbers[r]
            if curSum > target:
                r -= 1
            elif curSum < target:
                l += 1
            else:
                return [l + 1, r + 1]

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res = []
        for i, val in enumerate(nums):
            if val > 0: break
            if i > 0 and val == nums[i - 1]: continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                if val + nums[l] + nums[r] < 0:
                    l += 1
                elif val + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    res.append([val, nums[l], nums[r]])
                    l, r = l + 1, r - 1
                    while nums[l] == nums[l - 1] and l < r: l += 1
        return res

    def maxArea(self, height: list[int]) -> int:
        l, r, res = 0, len(height) - 1, 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            elif height[l] > height[r]:
                r -= 1
            if res >= (r - l) * max(height): break
        return res

    def trap(self, height: list[int]) -> int:
        # if not height: return 0
        # l, r, res = 0, len(height) - 1, 0
        # leftMax, rightMax = height[l], height[r]
        # while l < r:
        #     if leftMax < rightMax:
        #         l += 1
        #         leftMax = max(leftMax, height[l])
        #         res += leftMax - height[l]
        #     else:
        #         r -= 1
        #         rightMax = max(rightMax, height[r])
        #         res += rightMax - height[r]
        # return res
        maxl, maxr = 0, 0
        maxLeft = [maxl := max(maxl, h) for h in height]
        maxRight = [maxr := max(maxr, h) for h in reversed(height)]
        maxRight.reverse()
        return sum([min(maxLeft[i], maxRight[i]) - height[i] for i in range(len(height))])

    # new Picks from 300 list

    def validPalindrom(self, s: str) -> bool:
        # l, r = 0, len(s) -1 # <- better in Memory but not faster
        # def helper(l, r):
        #     while l < r:
        #         if s[l] == s[r]: l, r = l + 1, r - 1
        #         else: return False
        #     return True
        # while l < r:
        #     if s[l] == s[r]: l, r = l + 1, r - 1
        #     else: return helper(l+1, r) or helper(l, r-1)
        # return True
        mid = len(s) // 2 + 1
        s_rev = s[-mid:][::-1]
        if s[:mid] == s_rev: return True
        for i in range(0, mid):
            if s[i] != s_rev[i]:
                return (s[i + 1:mid] == s_rev[i:mid - 1] or
                        s[i:mid - 1] == s_rev[i + 1:])
        return True

    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        # nums.sort() <- slower but easier to implement
        # res = []
        # for i in range(len(nums) - 3):
        #     if i > 0 and nums[i] == nums[i - 1]: continue  # to avoid dups
        #     for j in range(i + 1, len(nums) - 2):
        #         if j > i + 1 and nums[j] == nums[j - 1]: continue
        #         l, r = j + 1, len(nums) - 1
        #         while l < r:
        #             if nums[i] + nums[j] + nums[l] + nums[r] < target: l += 1
        #             elif nums[i] + nums[j] + nums[l] + nums[r] > target: r -= 1
        #             else:
        #                 res.append([nums[i], nums[j], nums[l], nums[r]])
        #                 while l < r and nums[l] == nums[l + 1]: l += 1
        #                 while l < r and nums[r] == nums[r - 1]: r -= 1
        #                 l, r = l + 1, r - 1
        # return res
        nums.sort()
        res = []

        def findNSum(l, r, target, N, temp):
            if N > r - l + 1 or N < 2 or target < nums[l] * N or target > nums[r] * N: return
            if N == 2:
                while l < r:
                    if nums[l] + nums[r] < target:
                        l += 1
                    elif nums[l] + nums[r] > target:
                        r -= 1
                    else:
                        res.append(temp + [nums[l] + nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l - 1]: l += 1
            else:
                for i in range(l, r + 1):
                    if i == l or (i > l and nums[i - 1] != nums[i]):
                        findNSum(i + 1, r, target - nums[i], N - 1, temp + [nums[i]])

        findNSum(0, len(nums) - 1, target, 4, [])
        return res

    def mergeSortedArray(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        # for i in range(len(nums2)): <- less code
        #     bisect.insort_left(nums1, nums2[i], i, m + i)
        #     nums1.pop()
        while m > 0 and n > 0:
            if nums1[m - 1] >= nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
        if n > 0: nums1[:n] = nums2[:n]

    def reverseString(self, s: list[str]) -> None:
        l, r = 0, len(s) - 1
        while l < r: s[l], s[r], l, r = s[r], s[l], l + 1, r - 1

    def roateArray(self, nums: list[int], k: int) -> None:
        # l, r, k = 0, len(nums) - 1, k % len(nums)
        # def rotate(l, r):
        #     while l < r: nums[l], nums[r], l, r = nums[r], nums[l], l + 1, r - 1
        # rotate(l, r)
        # rotate(0, k - 1)
        # rotate(k, len(nums) - 1)
        k = k % len(nums)  # <- less code
        nums[:k], nums[k:] = nums[-k:], nums[:-k]

    def mergeAlternately(self, word1: str, word2: str) -> str:
        # res, i, j = "", 0, 0
        # while i < len(word1) and j < len(word2):
        #     res, i, j = res + word1[i] + word2[j], i + 1, j + 1
        # if i == len(word1): res += word2[j:]
        # if j == len(word2): res += word1[i:]
        # return res
        return "".join(a + b for a, b in zip(word1, word2)) + word1[len(word2):] + word2[len(word1):]

    def moveZeroes(self, nums: list[int]) -> None:
        z = 0
        for i in range(len(nums)):
            if nums[z] == 0 and nums[i] != 0: nums[i], nums[z] = nums[z], nums[i]
            if nums[z] != 0: z += 1

    def removeDuplicates(self, nums: list[int]) -> int:
        l = 1
        for r in range(1, len(nums)):
            if nums[r] != nums[r - 1]: nums[l], l = nums[r], l + 1
        return l

    def removeDuplicatesII(self, nums: list[int]) -> int:
        l, r = 0, 0
        while r < len(nums):
            count = 1
            while r + 1 < len(nums) and nums[r] == nums[r + 1]: r, count = r + 1, count + 1
            for i in range(min(2, count)):
                nums[l], l = nums[r], l + 1
            r += 1
        return l

    def numSubSeq(self, nums: list[int], target: int) -> int:
        nums.sort()
        res, mod, l, r = 0, (10 ** 9 + 7), 0, len(nums) - 1
        while l <= r:
            if (nums[l] + nums[r]) > target:
                r -= 1
            else:
                res, l = res + (1 << (r - l)), l + 1
        return res % mod

    #################### Sliding Window #################### 5/6 (2 HARD) - 5

    def max_profit(self, prices: list[int]) -> int:
        res, l = 0, 0
        for r in range(1, len(prices) - 1):
            if prices[l] > prices[r]: l = r
            res = max(res, prices[r] - prices[l])
        return res

    def length_of_longest_substring(self, s: str) -> int:
        l, res, charSet = 0, 0, set()
        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            res = max(res, r - l + 1)
        return res

    def character_replacement(self, s: str, k: int) -> int:
        l, maxl, res, mp = 0, 0, 0, {}
        for r in range(len(s)):
            mp[s[r]] = 1 + mp.get(s[r], 0)
            maxl = max(maxl, mp[s[r]])
            if (r - l + 1) - maxl > k:
                mp[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res

    def check_inclusion(self, s1: str, s2: str) -> bool:
        c1, c2, n = Counter(s1), Counter(), len(s1)
        for i, c in enumerate(s2):
            c2[c] += 1
            if i >= n: c2[s2[i - n]] -= 1
            if c1 == c2: return True
        return False

    def min_window_substring(self, s: str, t: str) -> str:
        if t == "": return ""
        countT, window = {}, {}
        for c in t: countT[c] = 1 + countT.get(c, 0)
        have, need = 0, len(countT)
        l, res, resLen = 0, [-1, -1], float("inf")
        for r in range(len(s)):
            window[s[r]] = 1 + window.get(s[r], 0)
            if s[r] in countT and window[s[r]] == countT[s[r]]: have += 1
            while have == need:  # Update our result
                if (r - l + 1) < resLen: res, resLen = [l, r], r - l + 1
                window[s[l]] -= 1  # Pop from the left of our window
                if s[l] in countT and window[s[l]] < countT[s[l]]: have -= 1
                l += 1
        l, r = res
        return s[l: r + 1] if resLen != float("inf") else ""

    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        res, q, l, r = [], collections.deque(), 0, 0
        while r < len(nums):
            # pop smaller values from q
            while q and nums[q[-1]] < nums[r]: q.pop()
            q.append(r)
            # Remove left val from window
            if l > q[0]: q.popleft()
            if (r + 1) >= k:
                res.append(nums[q[0]])
                l += 1
            r += 1
        return res

    # new Picks from 300 list

    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        cache = set()
        for i in range(len(nums)):
            if nums[i] in cache: return True
            cache.add(nums[i])
            if len(cache) == k + 1: cache.remove(nums[i - k])
        return False

    def numOfSubarrays(self, arr: list[int], k: int, threshold: int) -> int:
        res, curSum = 0, sum(arr[:k - 1])
        for i in range(len(arr) - k + 1):
            curSum += arr[i + k - 1]
            if (curSum / k) >= threshold: res += 1
            curSum -= arr[i]
        return res

    def maxFrequency(self, nums: list[int], k: int) -> int:
        nums.sort()
        curSum, l, res = nums[0], 0, 1
        for r in range(1, len(nums)):
            curSum += nums[r]
            while curSum + k < nums[r] * (r - l + 1):
                curSum -= nums[l]
                l += 1
            res = max(res, r - l + 1)
        return res

    def minSubArrayLen(self, nums: list[int], target: int) -> int:
        l, curSum, minlen = 0, 0, float('inf')
        for r in range(len(nums)):
            curSum += nums[r]
            while curSum >= target:
                minlen = min(minlen, r - l + 1)
                curSum -= nums[l]
                l += 1
        return minlen if minlen <= len(nums) else 0

    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        l, r = 0, len(arr) - k
        while l < r:
            m = (r + l) // 2
            if x - arr[m] > arr[m + k] - x:
                l = m + 1
            else:
                r = m
        return arr[l: l + k]

    def maxVowels(self, s: str, k: int) -> int:
        res, l, curr, charVow = 0, 0, 0, {'a', 'e', 'i', 'o', 'u'}
        for r in range(len(s)):
            if s[r] in charVow: curr += 1
            if (r - l + 1) > k:
                if s[l] in charVow: curr -= 1
                l += 1
            res = max(res, curr)
        return res

    #################### Stacks #################### 5/7 (1 HARD) - 5

    def isValid(self, s: str) -> bool:
        mp, stack = {")": "(", "]": "[", "}": "{"}, []
        for c in s:
            if c not in mp:
                stack.append(c)
                continue
            if not stack or stack[-1] != mp[c]: return False
            stack.pop()
        return not stack

    """
    155. Min Stack
    CHECK MinStack CLASS
    """

    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+" or c == "-" or c == "*" or c == "/":
                a, b = stack.pop(), stack.pop()
                if c == "+":
                    stack.append(b + a)
                elif c == "-":
                    stack.append(b - a)
                elif c == "/":
                    stack.append(int(b / a))
                elif c == "*":
                    stack.append(b * a)
            else:
                stack.append(int(c))
        return stack[0]

    def generateParenthesis(self, n: int) -> list[str]:
        stack, res = [], []

        def backtrack(openN, closedN):
            if openN == closedN == n:
                res.append("".join(stack))
                return
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            if closedN < openN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()

        backtrack(0, 0)
        return res

    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        res, stack = [0] * len(temperatures), []  # pair: [temp, index]
        for i, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = i - stackInd
            stack.append((t, i))
        return res

    def carFleet(self, target: int, position: list[int], speed: list[int]) -> int:
        pair, stack = [(p, s) for p, s in zip(position, speed)], []
        pair.sort(reverse=True)
        for p, s in pair:  # Reverse sorted order
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]: stack.pop()
        return len(stack)

    def largestRectangleArea(self, heights: list[int]) -> int:
        maxArea, stack = 0, []  # pair: (index, height)
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))
        for i, h in stack: maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea

    # new Picks from 300 list

    """
    225. MyStack
    CHECK MyStack CLASS
    """

    def calPoints(self, operations: list[str]) -> int:
        stack = []
        for op in operations:
            if op == "C" and len(stack) >= 1:
                stack.pop()
            elif op == "D" and len(stack) >= 1:
                temp = stack[-1] * 2
                stack.append(temp)
            elif op == "+" and len(stack) >= 2:
                temp = stack[-1] + stack[-2]
                stack.append(temp)
            else:
                stack.append(int(op))
        return sum(stack)

    def removeStars(self, s: str) -> str:
        stack = []
        for c in s:
            if c.isalpha():
                stack.append(c)
            elif c == "*":
                stack.pop()
        return "".join(stack)

    def nextStockSpanner(self, stack, price: int) -> int:
        span = 1
        while stack and stack[-1][0] <= price:
            span += stack[-1][1]
            stack.pop()
        stack.append((price, span))
        return span

    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        stack, j = [], 0
        for i in pushed:
            stack.append(i)
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        return len(stack) == 0

    #################### Binary Search #################### 5/7 (1 HARD) - 5

    def binarySearch(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) // 2)
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m
        return -1

    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        ROWS, COLS, l, r = len(matrix), len(matrix[0]), 0, len(matrix) * len(matrix[0]) - 1
        while l <= r:
            mid = (r + l) // 2
            row, col = divmod(mid, len(matrix[0]))
            if matrix[row][col] < target:
                l = mid + 1
            elif matrix[row][col] > target:
                r = mid - 1
            else:
                return True
        return False

    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        l, r, res = 1, max(piles), max(piles)
        while l <= r:
            m, time = (r + l) // 2, 0
            for p in piles: time += math.ceil(p / m)
            if time <= h:
                res, r = min(res, m), m - 1
            else:
                l = m + 1
        return res

    def findMin(self, nums: list[int]) -> int:
        l, r, cur_min = 0, len(nums) - 1, float('inf')
        while l <= r:
            mid = (r + l) // 2
            cur_min = min(cur_min, nums[mid])
            if nums[mid] > nums[r]:
                l = mid + 1  # right has the min
            else:
                r = mid - 1  # left has the min
        return min(cur_min, nums[l])

    def searchInRotatedSortedArray(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]: return mid
            if nums[l] <= nums[mid]:  # left sorted portion
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:  # right sorted portion
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1

    """
    981. Time Based Key-Value Store - TimeMap CLASS
    """

    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        i, j, merged, res = 0, 0, [], 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                merged.append(nums1[i])
                i += 1
            else:
                merged.append(nums2[j])
                j += 1
        if i < len(nums1): merged += nums1[i:]
        if j < len(nums2): merged += nums2[j:]
        m = len(merged) // 2
        return float(merged[m]) if len(merged) % 2 != 0 else (merged[m] + merged[m - 1]) / 2

    # new Picks from 300 list

    def searchInsert(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
        if r == mid:
            return mid + 1
        else:
            return mid

    def guessNumber(self, n: int) -> int:
        def guess(num: int) -> int:
            return num

        l, r = 1, n
        while True:
            mid = l + (r - l) // 2
            if guess(mid) == 0:
                return mid
            elif guess(mid) == 1:
                l = mid + 1
            else:
                r = mid - 1

    def singleNonDuplicate(self, nums: list[int]) -> int:
        # s = sum(nums) <--------- quick solution but its O(n)
        # o = sum(list(set(nums)))
        # return (2 * o - s)
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if mid < r and nums[mid] == nums[mid + 1]: mid += 1
            if nums[mid] != nums[mid - 1] and nums[mid] != nums[mid + 1]:
                return nums[mid]
            elif abs(mid - l) % 2 == 0:
                r = mid - 1
            else:
                l = mid + 1
        return nums[l]

    def searchInRotatedSortedArrayII(self, nums: list[int], target: int) -> bool:
        # return True if target in nums else False <----- one-liner works
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target: return True
            if nums[l] == nums[mid]:
                l += 1
                continue
            if nums[l] <= nums[mid]:
                if nums[l] <= target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] <= target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return False

    def minimizeMax(self, nums: list[int], p: int) -> int:
        nums.sort()
        l, r = 0, nums[len(nums) - 1] - nums[0]
        while l < r:
            mid = l + (r - l) // 2
            i, j = 1, 0
            while i < len(nums):
                if nums[i] - nums[i - 1] <= mid:
                    j += 1
                    i += 1
                i += 1
            if j >= p:
                r = mid
            else:
                l = mid + 1
        return l

    #################### LinkedList #################### 5/11 (2 HARD) - 5

    def reverseList(self, head: ListNode) -> ListNode:
        prev, cur = None, head
        while cur:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
        return prev

    def mergeTwoLinkedList(self, list1: ListNode, list2: ListNode) -> ListNode:
        dummy = ListNode()
        cur = dummy
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        if list1 or list2: cur.next = list1 if list1 else list2
        return dummy.next

    def hasCycle(self, head: ListNode) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast: return True
        return False

    def reorderLinkedList(self, head: ListNode) -> None:
        slow, fast = head, head.next  # find Middle
        while fast and fast.next: slow, fast = slow.next, fast.next.next
        second = slow.next  # reverse second half
        prev = slow.next = None
        while second:
            temp = second.next
            second.next = prev
            prev = second
            second = temp
        first, second = head, prev
        while second:
            temp1, temp2 = first.next, second.next
            first.next = second
            second.next = temp1
            first, second = temp1, temp2

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        left, right = dummy, head
        while n > 0: right, n = right.next, n - 1
        while right: right, left = right.next, left.next
        left.next = left.next.next
        return dummy.next

    def copyRandomList(self, head: ListNodeWithRandomPointer) -> ListNodeWithRandomPointer:
        mapp, cur = {None: None}, head
        while cur:
            copy = ListNodeWithRandomPointer(cur.val)
            mapp[cur] = copy
            cur = cur.next
        cur = head
        while cur:
            copy = mapp[cur]
            copy.next, copy.random = mapp[cur.next], mapp[cur.random]
            cur = cur.next
        return mapp[head]

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        cur, carry = dummy, 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            val = v1 + v2 + carry
            carry, val = divmod(val, 10)
            cur.next = ListNode(val)
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next

    def findDuplicate(self, nums: list[int]) -> int:
        slow, fast, slow2 = 0, 0, 0
        while True:
            slow, fast = nums[slow], nums[nums[fast]]
            if slow == fast: break
        while True:
            slow, slow2 = nums[slow], nums[slow2]
            if slow == slow2: return slow

    """
    146. LRU Cache - LRUCache Class
    """

    def mergeKLists(self, lists: list[ListNode]) -> ListNode:
        if not lists or len(lists) == 0: return None
        while len(lists) > 1:
            mergedLists = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                mergedLists.append(self.mergeTwoLinkedList(l1, l2))
            lists = mergedLists
        return lists[0]

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0, head)
        groupPrev = dummy
        while True:
            kth = self.getKthNode(groupPrev, k)
            if not kth: break
            groupNext = kth.next
            prev, cur = kth.next, groupPrev.next
            while cur != groupNext:
                temp = cur.next
                cur.next = prev
                prev, cur = cur, temp
            tmp = groupPrev.next
            groupPrev.next = kth
            groupPrev = tmp
        return dummy.next

    def getKthNode(self, curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr

    # new Picks from 300 list

    def isPalindromeLL(self, head: ListNode) -> bool:
        slow, fast = head, head.next
        while fast and fast.next: slow, fast = slow.next, fast.next.next
        second = slow.next
        prev = slow.next = None
        while second:
            temp = second.next
            second.next = prev
            prev = second
            second = temp
        first = head
        while first and prev:
            if first.val != prev.val: return False
            first, prev = first.next, prev.next
        return True

    def pairSumLL(self, head: ListNode) -> int:
        slow, fast, res = head, head.next, 0
        while fast and fast.next: slow, fast = slow.next, fast.next.next
        second = slow.next
        prev = slow.next = None
        while second:
            temp = second.next
            second.next = prev
            prev = second
            second = temp
        first = head
        while first and prev:
            res = max(res, int(first.val + prev.val))
            first, prev = first.next, prev.next
        return res
        # slow, fast, res, stack = head, head, 0, [] <------- alter solution with a list
        # while fast and fast.next:
        #     stack.append(slow)
        #     slow, fast = slow.next, fast.next.next
        # while slow:
        #     res, slow = max(res, slow.val + stack.pop().val), slow.next
        # return res

    """
    707. Design Linked List
    Check MyLinkedList CLASS
    """

    def partitionLL(self, head: ListNode, x: int) -> ListNode:
        dummy, temp = ListNode(0), ListNode(0)
        prev, curr = dummy, temp
        while head:
            if head.val < x:
                prev.next, prev = head, head
            else:
                curr.next, curr = head, head
            head = head.next
        curr.next = None
        prev.next = temp.next
        return dummy.next

    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummy = ListNode(0, head)
        leftPrev, cur, prev = dummy, head, None
        for _ in range(left - 1): leftPrev, cur = cur, cur.next  # 1) reach node at position "left"
        for _ in range(right - left + 1):  # 2) reverse from left to right
            temp = cur.next
            cur.next = prev
            prev, cur = cur, temp
        leftPrev.next.next = cur  # cur is node after "right"
        leftPrev.next = prev  # prev is "right"
        return dummy.next

    """
    622. Design Circular Queue
    Check MyCircularQueue CLASS
    """

    #################### Trees ######################### 5/15 (2 HARD) - 5

    def invertTree(self, root: TreeNode) -> TreeNode:
        # return root and TreeNode(root.val, self.invertTree(root.right), self.invertTree(root.left)) <- one liner
        if not root: return None
        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def maxDepth(self, root: TreeNode) -> int:
        # Recursive DFS
        return (1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
                if root else 0)
        # Iterative DFS
        # stack, res = [[root, 1]], 0
        # while stack:
        #     node, depth = stack.pop()
        #     if node:
        #         res = max(res, depth)
        #         stack.append([node.left, depth+1])
        #         stack.append([node.right, depth+1])
        # return res
        # BFS
        # q = deque()
        # if root:
        #     q.append(root)
        # level = 0
        # while q:
        #     for i in range(len(q)):
        #         node = q.popleft()
        #         if node.left():
        #             q.append(node.left)
        #         if node.right():
        #             q.append(node.right)
        #     level += 1
        # return level

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        res = 0

        def dfs(root):
            nonlocal res
            if not root: return 0
            left, right = dfs(root.left), dfs(root.right)
            res = max(res, left + right)
            return 1 + max(left, right)

        dfs(root)
        return res

    def isBalanced(self, root: TreeNode) -> bool:
        def height(root):  # <- getting just the height
            if not root: return 0
            left, right = height(root.left), height(root.right)
            if left < 0 or right < 0 or abs(left - right) > 1: return -1
            return 1 + max(left, right)

        return height(root) >= 0
        # def dfs(root):
        #     if not root: return [True, 0]
        #     left, right = dfs(root.left), dfs(root.right)
        #     balanced = left[0] and right[0] and abs(left[1]-right[1]) <=1
        #     return [balanced, 1 + max(left[1], right[1])]
        # return dfs(root)[0]

    def isSametree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q: return True
        if p and q and p.val == q.val:
            return self.isSametree(p.left, q.left) and self.isSametree(p.right, q.right)
        else:
            return False

    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if not subRoot: return True
        if not root: return False
        if self.isSametree(root, subRoot): return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def lowestCmmonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        cur = root
        while cur:
            if p.val < cur.val and q.val < cur.val:
                cur = cur.left
            elif p.val > cur.val and q.val > cur.val:
                cur = cur.right
            else:
                return cur

    def levelOrder(self, root: TreeNode) -> list[list[int]]:
        res, q = [], collections.deque()
        if root: q.append(root)
        while q:
            val = []
            for i in range(len(q)):
                node = q.popleft()
                val.append(node.val)
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            res.append(val)
        return res

    def rightSideView(self, root: TreeNode) -> list[int]:
        res, q = [], collections.deque([root])
        while q:
            rightSide = None
            for i in range(len(q)):
                node = q.popleft()
                if node:
                    rightSide = node
                    if node.left: q.append(node.left)
                    if node.right: q.append(node.right)
            if rightSide: res.append(rightSide.val)
        return res

    def goodNode(self, root: TreeNode) -> int:
        def dfs(root, maxVal) -> int:
            if not root: return 0
            res = 1 if root.val >= maxVal else 0
            maxVal = max(maxVal, root.val)
            res += dfs(root.left, maxVal)
            res += dfs(root.right, maxVal)
            return res

        return dfs(root, root.val)

    def isValidBST(self, root: TreeNode) -> bool:
        def valid(root, left, right) -> bool:
            if not root: return True
            if not (left < root.val < right): return False
            return valid(root.left, left, root.val) and valid(root.right, root.val, right)

        return valid(root, float("-inf"), float("-inf"))

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        cur, q = root, []
        while q or cur:
            while cur:
                q.append(cur)
                cur = cur.left
            cur, k = q.pop(), k - 1
            if k == 0: return cur.val
            cur = cur.right

    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        if not preorder or not inorder: return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root

    def maxPathSum(self, root: TreeNode) -> int:
        res = [root.val]

        def dfs(root):  # return max path sum without splitting
            if not root: return 0
            leftMax, rightMax = dfs(root.left), dfs(root.right)
            leftMax, rightMax = max(leftMax, 0), max(rightMax, 0)
            res[0] = max(res[0], root.val + leftMax + rightMax)  # WITH splitting
            return root.val + max(leftMax, rightMax)

        dfs(root)
        return res[0]

    def serialize(self, root: TreeNode):
        res = []

        def dfs(root):
            if not root:
                res.append("N")
                return
            res.append(str(root.val))
            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        vals = data.split(",")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left, node.right = dfs(), dfs()
            return node

        return dfs()

    # new Picks from 300 list
    def preOrderTraversal(self, root: TreeNode) -> list[int]:
        res = []

        def dfs(root):
            if not root: return
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return res

    def inorderTraversal(self, root: TreeNode) -> list[int]:
        res = []

        def dfs(root):
            if not root: return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)

        dfs(root)
        return res

    def postOrderTraversal(self, root: TreeNode) -> list[int]:
        res = []

        def dfs(root):
            if not root: return
            dfs(root.left)
            dfs(root.right)
            res.append(root.val)

        dfs(root)
        return res

    def sortedArrayToBST(self, nums: list[int]) -> TreeNode:
        if len(nums) == 0: return None
        root, mid = TreeNode(nums[len(nums) // 2]), len(nums) // 2
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])
        return root

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root: return TreeNode(val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)
        return root

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return None
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left: return root.right
            if not root.right: return root.left
            if root.left and root.right:
                node = root.right
                while node.left: node = node.left
                root.val = node.val
                root.right = self.deleteNode(root.right, root.val)
        return root

    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1 and not root2: return None
        if not root1 and root2: return root2
        if root1 and not root2: return root1
        temp = TreeNode(root1.val + root2.val)
        temp.left = self.mergeTrees(root1.left, root2.left)
        temp.right = self.mergeTrees(root1.right, root2.right)
        return temp

    def findDuplicateSubtrees(self, root: TreeNode) -> list[TreeNode]:
        res, seen = [], collections.defaultdict(int)

        def dfs(root):
            if not root: return
            node = tuple([dfs(root.left), root.val, dfs(root.right)])
            if node in seen and seen[node] == 1: res.append(root)
            seen[node] += 1
            return node

        dfs(root)
        return res

    #################### Tries ########################## 2/3 (1 HARD)

    """
    208. Implement Trie (Prefix Tree) - TRIE CLASS
    """

    """
    211. Design Add and Search Words in Data Structure - WordDictionary CLASS
    """

    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        root = TrieNode()
        for w in words: root.addWord(w)
        ROWS, COLS = len(board), len(board[0])
        res, visited = set(), set()

        def dfs(r, c, node, word):
            if (r not in range(ROWS) or
                    c not in range(COLS) or
                    board[r][c] not in node.children or
                    node.children[board[r][c]].refs < 1 or
                    (r, c) in visited): return
            visited.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.isWord:
                node.isWord = False
                res.add(word)
                root.removeWord(word)
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visited.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")
        return list(res)

    #################### Heap / Priority Queue ################ 5/7 (1 HARD) - 5

    """
    703. kth largest Element in a Stream - kthLargestElement CLASS
    """

    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-s for s in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            x, y = heapq.heappop(stones), heapq.heappop(stones)
            if y > x: heapq.heappush(stones, x - y)
        stones.append(0)
        return abs(stones[0])
        # stones = SortedList(stones)  # based on binary search and divide and conquer
        # while len(stones) > 1:
        #     y, x = stones.pop(), stones.pop()
        #     if x != y: stones.add(y - x)
        # return stones[0] if stones else 0

    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        minHeap = []
        for x, y in points:
            dist = -(x * x + y * y)
            if len(minHeap) == k:
                heapq.heappushpop(minHeap, (dist, x, y))
            else:
                heapq.heappush(minHeap, (dist, x, y))
        return [(x, y) for (dist, x, y) in minHeap]

    def findKthLargest(self, nums: list[int], k: int) -> int:
        # return sorted(nums, reverse=True)[k - 1] # <--- One-Liner O(nlogn)
        # if len(nums) == 1: return nums[0] # <--- very quick
        # return SortedList(nums)[-k]
        heap = nums[:k]  # <--- using Heap - O(nlogk)
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]: heapq.heappushpop(heap, num)
        return heap[0]

    def leastInterval(self, tasks: list[str], n: int) -> int:
        # count = Counter(tasks)
        # maxHeap = [-c for c in count.values()]
        # heapq.heapify(maxHeap)
        # time, q = 0, deque() # pair of [-c, idelTime]
        # while maxHeap or q:
        #     time += 1
        #     if not maxHeap: time = q[0][1]
        #     else:
        #         cnt = 1 + heapq.heappop(maxHeap)
        #         if cnt: q.append([cnt, time + n])
        #     if q and q[0][1] == time: heapq.heappush(maxHeap, q.popleft()[0])
        # return time
        values = list(Counter(tasks).values())
        max_val, ties = max(values), values.count(max(values))
        return max((max_val - 1) * (n * 2) + ties, len(tasks))

    """
    355. Design Twitter - Twitter CLASS
    """

    """
    295. Find Median from Data Stream - MedianFinder CLASS - HARD
    """

    # new Picks from 300 list

    def kthLargestNumber(self, nums: list[str], k: int) -> str:
        temp = [int(x) for x in nums]
        return str(SortedList(temp)[-k])

    def getOrder(self, tasks: list[list[int]]) -> list[int]:
        tasks = sorted([(task, i) for i, task in enumerate(tasks)])
        minHeap, res, prevTime = [], [], 0
        for (tt, pt), i in tasks:
            while minHeap and prevTime < tt:
                p, ind, ti = heapq.heappop(minHeap)
                prevTime = max(ti, prevTime) + p
                res.append(ind)
            heapq.heappush(minHeap, (pt, i, tt))
        return res + [i for _, i, _ in sorted(minHeap)]

    def maxScore(self, nums1: list[int], nums2: list[int], k: int) -> int:
        nums = sorted(((b, a) for a, b in zip(nums1, nums2)), reverse=True)
        heap = [num[1] for num in nums[:k]]
        ksum = sum(heap)
        heapq.heapify(heap)
        res = ksum * nums[k - 1][0]
        for num2, num1 in nums[k:]:
            ksum += num1 - heapq.heappushpop(heap, num1)
            res = max(res, ksum * num2)
        return res

    """
    1845. Seat Reservation Manager- SEATMANAGER CLASS
    """

    def assignTasks(self, servers: list[int], tasks: list[int]) -> list[int]:
        free, busy, res = [[weight, i, 0] for i, weight in enumerate(servers)], [], []
        heapq.heapify(free)
        for i, task in enumerate(tasks):
            while busy and busy[0][0] <= i or not free:
                time, weight, ind = heapq.heappop(busy)
                heapq.heappush(free, [weight, ind, time])
            weight, ind, time = heapq.heappop(free)
            res.append(ind)
            heapq.heappush(busy, [max(time, i) + task, weight, ind])
        return res

    #################### Backtracking #################### 6/9 (1 HARD) - 4

    def subsets(self, nums: list[int]) -> list[list[int]]:
        res, subsets = [], []

        def dfs(i):
            if i >= len(nums):
                res.append(subsets.copy())
                return
            subsets.append(nums[i])  # decision to include nums[i]
            dfs(i + 1)
            subsets.pop()  # decision NOT to include nums[i]
            dfs(i + 1)

        dfs(0)
        return res
        # res = [[]] <- much faster O(nlogn)
        # for i in range(1, len(sorted(nums)) + 1):
        #     subset = itertools.combinations(nums, i)
        #     for val in subset: res.append(list(val))
        # return res

    def subsetsIIWithDup(self, nums: list[int]) -> list[list[int]]:
        # res = []
        # nums.sort()
        # def dfs(i, subset):
        #     if i == len(nums):
        #         res.append(subset[::])
        #         return
        #     subset.append(nums[i]) # All subsets that includes nums[i]
        #     dfs(i+1, subset)
        #     subset.pop()
        #     while i + 1 < len(nums) and nums[i] == nums[i+1]: i += 1
        #     dfs(i+1, subset)
        # dfs(0, [])
        # return res
        res = []  # <------ less code checks if subset exists but still creates it
        nums.sort()

        def dfs(i, subset):
            if i == len(nums):
                res.append(subset) if subset not in res else None
                return
            dfs(i + 1, subset + [nums[i]])  # pick
            dfs(i + 1, subset)  # not pick

        dfs(0, [])
        return res

    def permute(self, nums: list[int]) -> list[list[int]]:
        # return permutation(nums) <------ ONE LINER
        # res = []
        # if len(nums) == 1: return [nums[:]] # base case
        # for i in range(len(nums)):
        #     temp = nums.pop(0)
        #     perms = self.permute(nums)
        #     for perm in perms: perm.append(temp)
        #     res.extend(perms)
        #     nums.append(temp)
        # return res
        res = []  # <-- slightly faster

        def dfs(nums, subs):
            if len(nums) == 0:
                res.append(subs)
                return
            for i in range(len(nums)):
                dfs(nums[:i] + nums[i + 1:], subs + [nums[i]])

        dfs(nums, [])
        return res

    def isPali(self, s, l, r):
        while l < r:
            if s[l] != s[r]: return False
            l, r = l + 1, r - 1
        return True

    def palindromePartitioning(self, s: str) -> list[list[str]]:
        res, part = [], []

        def dfs(i):
            if i >= len(s):
                res.append(part.copy())
                return
            for j in range(i, len(s)):
                if self.isPali(s, i, j):
                    part.append(s[i: j + 1])
                    dfs(j + 1)
                    part.pop()

        dfs(0)
        return res

    def letterCombinations(self, digits: str) -> list[str]:
        res = []
        digitToChar = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz"
        }

        def dfs(i, curStr):
            if len(curStr) == len(digits):
                res.append(curStr)
                return
            for c in digitToChar[digits[i]]: dfs(i + 1, curStr + c)

        if digits: dfs(0, "")
        return res

    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        # res, subset = [], []
        # def dfs(i, sum):
        #     if sum == target:
        #         res.append(subset.copy())
        #         return
        #     if i >= len(candidates) or sum > target: return
        #     subset.append(candidates[i])
        #     dfs(i, sum + candidates[i])
        #     subset.pop()
        #     dfs(i+1, sum)
        # dfs(0,0)
        # return res
        res = []  # Faster

        def dfs(candidates, target, subsets):
            if target == 0:
                res.append(subsets)
                return
            for i in range(len(candidates)):
                if candidates[i] > target: continue
                dfs(candidates[i:], target - candidates[i], subsets + [candidates[i]])

        dfs(candidates, target, [])
        return res

    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        # candidates.sort()
        # res = []
        # def dfs(pos, cur, target):
        #     if target == 0:
        #         res.append(cur.copy())
        #         return
        #     if target <= 0: return
        #     prev = -1
        #     for i in range(pos, len(candidates)):
        #         if candidates[i] == prev: continue
        #         cur.append(candidates[i])
        #         dfs(i+1, cur, target - candidates[i])
        #         cur.pop()
        #         prev = candidates[i]
        # dfs(0, [], target)
        # return res
        res = []  # <-- slightly faster and less code
        candidates.sort()

        def dfs(candidates, target, subset):
            if target == 0:
                res.append(subset)
                return
            for i in range(len(candidates)):
                if candidates[i] > target: continue
                if i >= 1 and candidates[i] == candidates[i - 1]: continue
                dfs(candidates[i + 1:], target - candidates[i], subset + [candidates[i]])

        dfs(candidates, target, [])
        return res

    def wordSearchExist(self, board: list[list[str]], word: str) -> bool:
        ROWS, COLS, seenPath = len(board), len(board[0]), set()

        def dfs(r, c, i):
            if i == len(word): return True
            if (min(r, c) < 0 or r >= ROWS or c >= COLS or word[i] != board[r][c] or
                    (r, c) in seenPath): return False
            seenPath.add((r, c))
            res = dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1) or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1)
            seenPath.remove((r, c))
            return res

        # To prevent TLE, reverse the word if frequency of the first letter is more than the last letter's
        count = defaultdict(int, sum(map(Counter, board), Counter()))
        if count[word[0]] > count[word[-1]]: word = word[::-1]
        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0): return True
        return False

    def solveNQueens(self, n: int) -> list[list[str]]:
        res, col, posDiag, negDiag = [], set(), set(), set()  # pos:(r+c), neg:(r-c)
        board = [["."] * n for i in range(n)]

        def dfs(r):
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return
            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag: continue
                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                board[r][c] = "Q"
                dfs(r + 1)
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
                board[r][c] = "."

        dfs(0)
        return res

    # new Picks from 300 list

    def combine(self, n: int, k: int):
        # return itertools.combinations(range(1, n+1), k) <------- ONE LINER
        # nums, subsets = [*range(1, n + 1)], set()  # <- short but not efficient
        # for _ in range(n):
        #     subs = itertools.combinations(nums, k)
        #     for val in subs: subsets.add(val)
        # return [list(i) for i in subsets]
        res = []

        def dfs(first, subset):
            if len(subset) == k:
                res.append(subset[:])
                return
            for i in range(first, n + 1):
                subset.append(i)
                dfs(i + 1, subset)
                subset.pop()

        dfs(1, [])
        return res

    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        # output = set(val for val in itertools.permutations(nums))
        # return [list(val) for val in output] # <------- ONE LINER
        res, count, perm = [], Counter(nums), []  # <- much faster

        def dfs():
            if len(perm) == len(nums):
                res.append(perm[::])
            for i in count:
                if count[i] == 0: continue
                perm.append(i)
                count[i] -= 1
                dfs()
                perm.pop()
                count[i] += 1

        dfs()
        return res

    def restoreIPAddresses(self, s: str) -> list[str]:
        # res = [] # <--- Recursive
        # def dfs(s, k, temp):
        #     if k == 4 and len(s) == 0:
        #         res.append(temp[:-1])
        #         return
        #     if k == 4 or len(s) == 0: return
        #     for i in range(3):
        #         if k > 4 or i + 1 > len(s): break
        #         if int(s[:i + 1]) > 255: continue
        #         if i != 0 and s[0] == "0": continue
        #         dfs(s[i + 1:], k + 1, temp + s[:i + 1] + '.')
        # dfs(s, 0, '')
        # return res
        points = ((i, j, k)  # <--- much faster iretive solution
                  for i in range(0 + 1, min(0 + 4, len(s)))
                  for j in range(i + 1, min(i + 4, len(s)))
                  for k in range(j + 1, min(j + 4, len(s))))
        is_valid_ip = lambda ip: all(map(lambda x: int(x) <= 255 and (len(x) == 1 or x[0] != '0'), ip))
        candidate_ips = ((s[:i], s[i:j], s[j:k], s[k:]) for i, j, k in points)
        valid_ips = filter(is_valid_ip, candidate_ips)
        return ['.'.join(ip) for ip in valid_ips]

    def makesquare(self, matchsticks: list[int]) -> bool:
        target, m = divmod(sum(matchsticks), 4)
        if m: return False
        targetLst = [0] * 4
        matchsticks.sort(reverse=True)

        def dfs(i):
            if i == len(matchsticks): return len(set(targetLst)) == 1
            for j in range(4):
                if matchsticks[i] + targetLst[j] > target: continue
                targetLst[j] += matchsticks[i]
                if dfs(i + 1): return True
                targetLst[j] -= matchsticks[i]
                if not targetLst[j]: break
            return False

        return matchsticks[0] <= target and dfs(0)

    #################### Graphs #################### 10/13 (1 HARD) - 5

    """
    For DFS we add the visited set if we cant modify the original grid
    """

    def numIslands(self, grid: list[list[str]]) -> int:
        # def bfs(r, c):
        #     q = deque()
        #     visited.add((r, c))
        #     q.append((r, c))
        #     while q:
        #         row, col = q.popleft()
        #         directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        #         for dr, dc in directions:
        #             r, c = row + dr, col + dc
        #             if (c in range(rows) and c in range(cols) and
        #                 grid[r][c] == "1" and (r, c) not in visited):
        #                 q.append((r, c))
        #                 visited.add((r, c))
        # ROWS, COLS, islands = len(grid), len(grid[0]), 0 # <- cant get any faster
        # def dfs(r, c):
        #     grid[r][c] = - 1
        #     if r < ROWS - 1 and grid[r + 1][c] == "1":  dfs(r + 1, c)
        #     if r > 0 and grid[r - 1][c] == "1":         dfs(r - 1, c)
        #     if c < COLS - 1 and grid[r][c + 1] == "1":  dfs(r, c + 1)
        #     if c > 0 and grid[r][c - 1] == "1":         dfs(r, c - 1)
        # for r in range(ROWS):
        #     for c in range(COLS):
        #         if grid[r][c] == "1":
        #             islands += 1
        #             dfs(r, c)
        if not grid or len(grid[0]) == 0: return 0
        ROWS, COLS, visited, islands = len(grid), len(grid[0]), set(), 0

        def dfs(r, c):
            if (r not in range(ROWS) or c not in range(COLS) or
                    grid[r][c] == "0" or (r, c) in visited): return
            visited.add((r, c))
            for dr, dc in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                dfs(r + dr, c + dc)

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == "1" and (r, c) not in visited:
                    islands += 1
                    dfs(r, c)
        return islands

    def cloneGraph(self, node: Node) -> Node:
        # if not node: return node # <- BFS
        # q, clone = deque([node]), {node.val: Node(node.val, [])}
        # while q:
        #     cur = q.popleft()
        #     cur_clone = clone[cur.val]
        #     for chil in cur.neighbors:
        #         if chil.val not in clone:
        #             clone[chil.val] = Node(chil.val, [])
        #             q.append(chil)
        #         cur_clone.neighbors.append(clone[chil.val])
        # return clone[node.val]
        if not node: return node
        oldToNew = {}  # <- DFS

        def dfs(node):
            if node in oldToNew: return oldToNew[node]
            copy = Node(node.val)
            oldToNew[node] = copy
            for neighbor in node.neighbors:
                copy.neighbors.append(dfs(neighbor))
            return copy

        return dfs(node) if node else None

    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        if not grid or len(grid[0]) == 0: return 0
        ROWS, COLS, res = len(grid), len(grid[0]), 0

        def dfs(r, c):
            if 0 > r >= ROWS or 0 > c >= COLS or grid[r][c] == 0: return 0
            grid[r][c] = 0
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1: res = max(res, dfs(r, c))
        return res

    def pacificAtlantic(self, heights: list[list[int]]) -> list[list[int]]:
        # ROWS, COLS = len(heights), len(heights[0]) # <- alt solution not faster tho
        # isPacc = [[True if i == 0 or j == 0 else False for j in range(COLS)] for i in range(ROWS)]
        # isAtl = [[True if i == ROWS - 1 or j == COLS - 1 else False for j in range(COLS)] for i in range(ROWS)]
        # def dfs(i, j, isPac):
        #     val = heights[i][j]
        #     for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
        #         if 0 <= x < ROWS and 0 <= y < COLS and not isPac[x][y] and heights[x][y] >= val:
        #             isPac[x][y] = True
        #             dfs(x, y, isPac)
        # for i in range(ROWS): dfs(i, 0, isPacc)
        # for j in range(COLS): dfs(0, j, isPacc)
        # for i in range(ROWS): dfs(i, COLS - 1, isAtl)
        # for j in range(COLS): dfs(ROWS - 1, j, isAtl)
        # return [[i, j] for i in range(ROWS) for j in range(COLS) if isPacc[i][j] and isAtl[i][j]]
        if not heights or not len(heights): return heights
        ROWS, COLS, res, pac, atl = len(heights), len(heights[0]), [], set(), set()

        def dfs(r, c, visit, prevh):
            if r < 0 or r == ROWS or c < 0 or c == COLS or heights[r][c] < prevh or (r, c) in visit: return
            visit.add((r, c))
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        for c in range(COLS):
            dfs(0, c, pac, heights[0][c])
            dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])
        for r in range(ROWS):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, COLS - 1, atl, heights[r][COLS - 1])
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])
        return res

    def solveSurroundedRegions(self, board: list[list[str]]) -> None:
        # if len(board) < 3 or len(board[0]) < 3: return
        # ROWS, COLS = len(board), len(board[0])
        # def dfs(r, c):
        #     board[r][c] = 'R'
        #     if r > 0 and board[r-1][c] == 'O':         dfs(r - 1, c)
        #     if r < ROWS - 1 and board[r+1][c] == 'O':  dfs(r + 1, c)
        #     if c > 0 and board[r][c-1] == 'O':         dfs(r, c - 1)
        #     if c < COLS - 1 and board[r][c+1] == 'O':  dfs(r, c + 1)
        # for r in range(ROWS):
        #     if board[r][0] == 'O':        dfs(r, 0)
        #     if board[r][COLS - 1] == 'O': dfs(r, COLS - 1)
        # for c in range(1, COLS - 1):
        #     if board[0][c] == 'O':        dfs(0, c)
        #     if board[ROWS - 1][c] == 'O': dfs(ROWS - 1, c)
        # for r in range(ROWS):
        #     for c in range(COLS):
        #         if board[r][c] == 'O':   board[r][c] = 'X'
        #         elif board[r][c] == 'R': board[r][c] = 'O'
        ROWS, COLS = len(board), len(board[0])

        def dfs(r, c):
            if r < 0 or r == ROWS or c < 0 or c == COLS or board[r][c] != "O": return
            board[r][c] = "T"
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        # 1. (DFS) Capture unsurrounded regions (O -> T)
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O" and (r in [0, ROWS - 1] or c in [0, COLS - 1]):
                    dfs(r, c)
        # 2. Capture surrounded regions (O -> x)
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O": board[r][c] = "X"
        # 3. Uncapture unsurroounded regions (T -> O)
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "T": board[r][c] = "O"

    """
    994. Rotting Oranges
    You are given an m x n grid where each cell can have one of three values:
    0 representing an empty cell,
    1 representing a fresh orange, or
    2 representing a rotten orange.
    Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
    Return the minimum number of minutes that must elapse until no cell has a fresh orange. 
    If this is impossible, return -1.
    Ex1: Input: grid = [[2,1,1],[1,1,0],[0,1,1]], Output: 4
    Ex2: Input: grid = [[2,1,1],[0,1,1],[1,0,1]], Output: -1
    Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, 
    because rotting only happens 4-directionally.
    Ex3: Input: grid = [[0,2]], Output: 0
    Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
    BFS solution
    """

    def orangesRotting(self, grid: list[list[int]]) -> int:
        if not grid or not len(grid): return 0
        ROWS, COLS, time, fresh, q = len(grid), len(grid[0]), 0, 0, collections.deque()
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1: fresh += 1
                if grid[r][c] == 2: q.append((r, c))
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while fresh > 0 and q:
            for i in range(len(q)):
                row, col = q.popleft()
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if r in range(ROWS) and c in range(COLS) and grid[r][c] == 1:
                        grid[r][c] = 2
                        q.append((r, c))
                        fresh -= 1
            time += 1
        return time if fresh == 0 else -1

    """
    663. Walls and Gates
    You are given a m x n 2D grid initialized with these three possible values.
    -1 - A wall or an obstacle.
    0 - A gate.
    INF - Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 to represent INF 
    as you may assume that the distance to a gate is less than 2147483647.
    Fill each empty room with the distance to its nearest gate. 
    If it is impossible to reach a Gate, that room should remain filled with INF
    Ex1: Input: [[2147483647,-1,0,2147483647],
                 [2147483647,2147483647,2147483647,-1],
                 [2147483647,-1,2147483647,-1],
                 [0,-1,2147483647,2147483647]]
        Output:
        [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]
    Explanation:
        the 2D grid is:
        INF  -1  0  INF
        INF INF INF  -1
        INF  -1 INF  -1
          0  -1 INF INF
        the answer is:
          3  -1   0   1
          2   2   1  -1
          1  -1   2  -1
          0  -1   3   4
    Ex2: Input: [[0,-1],[2147483647,2147483647]], Output: [[0,-1],[1,2]]
    """

    def wallsAndGates(self, rooms: list[list[int]]):
        ROWS, COLS, visited, q = len(rooms), len(rooms[0]), set(), deque()

        def addRoom(r, c):
            if (min(r, c) < 0 or r == ROWS or c == COLS or (r, c) in visited or rooms[r][c] == -1):
                return
            visited.add((r, c))
            q.append((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                if rooms[r][c] == 0:
                    q.append((r, c))
                    visited.add((r, c))
        dist = 0
        while q:
            for i in range(len(q)):
                r, c = q.popleft()
                rooms[r][c] = dist
                addRoom(r + 1, c)
                addRoom(r - 1, c)
                addRoom(r, c + 1)
                addRoom(r, c - 1)
            dist += 1

    """
    207. Course Schedule
    There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
    You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that 
    you must take course bi first if you want to take course ai.
    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    Return true if you can finish all courses. Otherwise, return false.
    Ex1: Input: numCourses = 2, prerequisites = [[1,0]], Output: true
    Explanation: There are a total of 2 courses to take. 
    To take course 1 you should have finished course 0. So it is possible.
    Ex2: Input: numCourses = 2, prerequisites = [[1,0],[0,1]], Output: false
    Explanation: There are a total of 2 courses to take. 
    To take course 1 you should have finished course 0, 
    and to take course 0 you should also have finished course 1. So it is impossible.
    Ex3: Input: numCourses = 5, prerequisites = [[0,1],[0,2],[1,3],[1,4],[3,4]], Output: true
    """

    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        preMap = {c: [] for c in range(numCourses)}
        for crs, pre in prerequisites:
            preMap[crs].append(pre)
        visit = set()

        def dfs(crs):
            if crs in visit: return False
            if preMap[crs] == []: return True
            visit.add(crs)
            for pre in preMap[crs]:
                if not dfs(pre): return False
            visit.remove(crs)
            preMap[crs] = []
            return True

        for c in range(numCourses):
            if not dfs(c): return False
        return True

    """
    210. Course Schedule II
    There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
    You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that 
    you must take course bi first if you want to take course ai.
    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    Return the ordering of courses you should take to finish all courses. 
    If there are many valid answers, return any of them. 
    If it is impossible to finish all courses, return an empty array.
    Ex1: Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]], Output: [0,2,1,3]
    Explanation: There are a total of 4 courses to take. 
        To take course 3 you should have finished both courses 1 and 2. 
        Both courses 1 and 2 should be taken after you finished course 0.
        So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
    Ex2: Input: numCourses = 2, prerequisites = [[1,0]], Output: [0,1]
    Explanation: There are a total of 2 courses to take. 
        To take course 1 you should have finished course 0. So the correct course order is [0,1].
    Ex3: Input: numCourses = 1, prerequisites = [], Output: [0]
    """

    def findOrder(self, numCourses: int, prerequisites: list[list[int]]) -> list[int]:
        preMap = {c: [] for c in range(numCourses)}
        for crs, pre in prerequisites:
            preMap[crs].append(pre)
        visit, cycle, res = set(), set(), []

        def dfs(crs):
            if crs in cycle: return False
            if crs in visit: return True
            cycle.add(crs)
            for pre in preMap[crs]:
                if dfs(pre) == False: return False
            cycle.remove(crs)
            visit.add(crs)
            res.append(crs)
            return True

        for c in range(numCourses):
            if dfs(c) == False: return []
        return res

    """
    684. Redundant Connection
    In this problem, a tree is an undirected graph that is connected and has no cycles.
    You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. 
    The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. 
    The graph is represented as an array edges of length n where edges[i] = [ai, bi] 
    indicates that there is an edge between nodes ai and bi in the graph.
    Return an edge that can be removed so that the resulting graph is a tree of n nodes. 
    If there are multiple answers, return the answer that occurs last in the input.
    Ex1: Input: edges = [[1,2],[1,3],[2,3]], Output: [2,3]
    Ex2: Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]], Output: [1,4]
    UNION FIND ALGO
    """

    def findRedundantConnection(self, edges: list[list[int]]) -> list[int]:
        par = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)

        def find(n):
            p = par[n]
            while p != par[p]:
                par[p] = par[par[p]]
                p = par[p]
            return p

        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2: return False
            if rank[p1] > rank[p2]:
                par[p2] = p1
                rank[p1] += rank[p2]
            else:
                par[p1] = p2
                rank[p2] += rank[p1]
            return True

        for n1, n2 in edges:
            if not union(n1, n2): return [n1, n2]

    """
    591. Connecting Graph III
    Given n nodes in a graph, denoted 1 through n. ConnectingGraph3(n) creates n nodes, 
    and at the beginning there are no edges in the graph.
    You need to support the following method:
    connect(a, b), an edge to connect node a and node b
    query(), Returns the number of connected component in the graph
    """

    def countComponents(self, n: int, edges: list[list[int]]) -> int:
        dsu = UnionFind()
        for a, b in edges:
            dsu.union(a, b)
        return len(set(dsu.find_parent(x) for x in range(n)))

    """
    178. Graph Valid Tree
    Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
    write a function to check whether these edges make up a valid tree.
    Ex1: Input: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]], Output: true.
    Ex2: Input: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], Output: false.
    """

    def validTree(self, n, edges):
        if not n: return True
        adj, visit = {i: [] for i in range(n)}, set()
        for n1, n2 in edges:
            adj[n1].append(n2)
            adj[n2].append(n1)

        def dfs(i, prev):
            if i in visit: return False
            visit.add(i)
            for j in adj[i]:
                if j == prev: continue
                if not dfs(j, i): return False
            return True

        return dfs(0, -1) and n == len(visit)

    """
    127. Word Ladder - HARD
    A transformation sequence from word beginWord to word endWord using a 
    dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
    Every adjacent pair of words differs by a single letter.
    Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
    sk == endWord
    Given two words, beginWord and endWord, and a dictionary wordList, 
    return the number of words in the shortest transformation sequence from 
    beginWord to endWord, or 0 if no such sequence exists.
    Ex1: Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"], Output: 5
    Explanation: 
        One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", 
        which is 5 words long.
    Ex2: Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"], Output: 0
    Explanation: 
        The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
    """

    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        if endWord not in wordList: return 0
        nei = collections.defaultdict(list)
        wordList.append(beginWord)
        for word in wordList:
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j + 1:]
                nei[pattern].append(word)
        visit, q, res = set([beginWord]), deque([beginWord]), 1
        while q:
            for i in range(len(q)):
                word = q.popleft()
                if word == endWord: return res
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j + 1:]
                    for neiWord in nei[pattern]:
                        if neiWord not in visit:
                            visit.add(neiWord)
                            q.append(neiWord)
            res += 1
        return 0

    # new Picks from 300 list

    def islandPerimeter(self, grid: list[list[int]]) -> int:
        # ROWS, COLS, visit = len(grid), len(grid[0]), set()
        # def dfs(r, c):
        #     if 0 > r or r >= ROWS or 0 > c or c >= COLS or grid[r][c] == 0: return 1
        #     if (r, c) in visit: return 0
        #     visit.add((r, c))
        #     res = dfs(r + 1, c)
        #     res += dfs(r - 1, c)
        #     res += dfs(r, c + 1)
        #     res += dfs(r, c - 1)
        #     return res
        # for r in range(ROWS):
        #     for c in range(COLS):
        #         if grid[r][c]: return dfs(r, c)
        row, col = len(grid), len(grid[0])

        def dfs(r, c) -> int:
            return ((1 if c == 0 else grid[r][c - 1] == 0) +
                    (1 if r == 0 else grid[r - 1][c] == 0) +
                    (1 if c == col - 1 else grid[r][c + 1] == 0) +
                    (1 if r == row - 1 else grid[r + 1][c] == 0))

        return sum(dfs(r, c) for r in range(row) for c in range(col) if grid[r][c] == 1)

    def isAlienSorted(self, words: list[str], order: str) -> bool:
        # return [[chr(order.index(word[i])) for i in range(len(word))] for word in words] == sorted(
        #     [[chr(order.index(word[i])) for i in range(len(word))] for word in words]) #<- one-liner
        cache = {c: i for i, c in enumerate(order)}

        def dfs(word1, word2):
            i, j = 0, 0
            while i < len(word1) and j < len(word2):
                if cache[word1[i]] > cache[word2[j]]:
                    return False
                elif cache[word1[i]] < cache[word2[j]]:
                    return True
                i, j = i + 1, j + 1
            return len(word1) <= len(word2)

        for i in range(len(words) - 1):
            if not dfs(words[i], words[i + 1]): return False
        return True

    def countSubIslands(self, grid1: list[list[int]], grid2: list[list[int]]) -> int:
        ROWS, COLS, visit, res = len(grid1), len(grid1[0]), set(), 0  # <- DFS

        def dfs(r, c):
            if 0 > r or r == ROWS or 0 > c or c == COLS or (r, c) in visit or grid2[r][c] == 0: return True
            visit.add((r, c))
            island = True
            if grid1[r][c] == 0: island = False
            island = dfs(r + 1, c) and island
            island = dfs(r - 1, c) and island
            island = dfs(r, c + 1) and island
            island = dfs(r, c - 1) and island
            return island

        for r in range(ROWS):
            for c in range(COLS):
                if grid2[r][c] and (r, c) not in visit and dfs(r, c):
                    res += 1
        return res

    def minReorder(self, n: int, connections: list[list[int]]) -> int:
        graph = defaultdict(list)

        def dfs(node, parent) -> int:
            return sum(cost + dfs(kid, node) for kid, cost in graph[node] if kid != parent)

        for u, v in connections:
            graph[u].append((v, 1))
            graph[v].append((u, 0))
        return dfs(0, -1)

    def openLock(self, deadends: list[str], target: str) -> int:
        if "0000" == target: return 0
        if "0000" in deadends: return -1
        start, end, stop, steps = {"0000"}, {target}, set(deadends), 0
        neighbors = {str(i): (str((i - 1) % 10), str((i + 1) % 10)) for i in range(10)}
        while start and end:
            steps += 1
            if len(start) > len(end): start, end = end, start
            temp = set()
            for lock in start:
                stop.add(lock)
                for i in range(4):
                    for n in neighbors[lock[i]]:
                        newLock = lock[:i] + n + lock[i + 1:]
                        if newLock in end: return steps
                        if newLock in stop: continue
                        temp.add(newLock)
            start = temp
        return -1

    #################### Advance Graphs #################### /6 (3 HARD) -

    def minCostConnectPoints(self, points: list[list[int]]) -> int:
        # N = len(points)
        # adj = {i: [] for i in range(N)} # i : list of [cost, node]
        # for i in range(N):
        #     x1, y1 = points[i]
        #     for j in range(i+1, N):
        #         x2, y2 = points[j]
        #         dist = abs(x1 - x2) + abs(y1 - y2)
        #         adj[i].append([dist, j])
        #         adj[j].append([dist, i])
        # # Prim's
        # res, visit = 0, set()
        # minH = [[0, 0]] # [cost, point]
        # while len(visit) < N:
        #     cost, i = heapq.heappop(minH)
        #     if i in visit: continue
        #     res += cost
        #     visit.add(i)
        #     for neiCost, nei in adj[i]:
        #         if nei not in visit:
        #             heapq.heappush(minH, [neiCost, nei])
        # return res
        dist, res = {(x, y): float('inf') if i else 0 for i, (x, y) in enumerate(points)}, 0
        while dist:  # <- much faster
            x, y = min(dist, key=dist.get)  # obtain the current min edge
            res += dist.pop((x, y))  # remove the point
            for x1, y1 in dist:
                dist[(x1, y1)] = min(dist[x1, y1], abs(x - x1) + abs(y - y1))
        return res

    def networkDelayTime(self, times: list[list[int]], n: int, k: int) -> int:
        edges, minH, visit, time = collections.defaultdict(list), [(0, k)], set(), 0
        for u, v, w in times: edges[u].append((v, w))
        while minH:
            w1, n1 = heapq.heappop(minH)
            if n1 in visit: continue
            visit.add(n1)
            time = w1
            for n2, w2 in edges[n1]:
                if n2 not in visit: heapq.heappush(minH, (w1 + w2, n2))
        return time if len(visit) == n else -1

    def findCheapestPrice(self, n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0
        for _ in range(k + 1):
            tmpPrices = prices.copy()
            for s, d, p in flights:  # s = source, d = destination, p = price
                if prices[s] == float("inf"): continue
                if prices[s] + p < tmpPrices[d]: tmpPrices[d] = prices[s] + p
            prices = tmpPrices
        return prices[dst] if prices[dst] != float("inf") else -1

    def maxProbability(self, n: int, edges: list[list[int]], succProb: list[float],
                       start_node: int, end_node: int) -> float:
        adj, pq, visit = collections.defaultdict(list), [(-1, start_node)], set()
        for i in range(len(edges)):
            src, dst = edges[i]
            adj[src].append([dst, succProb[i]])
            adj[dst].append([src, succProb[i]])
        while pq:
            prob, cur = heapq.heappop(pq)
            visit.add(cur)
            if cur == end_node: return prob * -1
            for nei, edgeProb in adj[cur]:
                if nei not in visit: heapq.heappush(pq, (prob * edgeProb, nei))
        return 0

    def findItinetrary(self, tickets: list[list[str]]) -> list[str]:
        # adj = {src : [] for src , dst in tickets}
        # res = ["JFK"]
        # tickets.sort()
        # for src, dst in tickets: adj[src].append(dst)
        # def dfs(src):
        #     if len(res) == len(tickets) + 1: return True
        #     if src not in adj: return False
        #     temp = list(adj[src])
        #     for i, v in enumerate(temp):
        #         adj[src].pop(i)
        #         res.append(v)
        #         if dfs(v): return True
        #         adj[src].insert(i, v)
        #         res.pop()
        #     return False
        # dfs("JFK")
        # return res
        graph, stack, res = collections.defaultdict(list), ["JFK"], []
        for src, dst in tickets: graph[src].append(dst)  # <- faster and a bit more efficient
        for src in graph.keys(): graph[src].sort(reverse=True)
        while stack:
            city = stack[-1]
            if city in graph and len(graph[city]) > 0:
                stack.append(graph[city].pop())
            else:
                res.append(stack.pop())
        return res[::-1]

    def swimInWater(self, grid: list[list[int]]) -> int:
        ROWS, COLS, visit, pq = len(grid), len(grid[0]), set(), [[grid[0][0], 0, 0]]  # (max(time/height), r, c)
        visit.add((0, 0))
        while pq:
            time, row, col = heapq.heappop(pq)
            if row == ROWS - 1 and col == COLS - 1: return time
            for dr, dc in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                r, c = row + dr, col + dc
                if 0 > r or r == ROWS or 0 > c or c == COLS or (r, c) in visit: continue
                visit.add((r, c))
                heapq.heappush(pq, [max(time, grid[r][c]), r, c])

    def alienOrder(self, words: list[str]) -> str:
        adj, visited, res = {char: set() for word in words for char in word}, {}, []  # false=visited, true=currentPath
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            minLen = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]: return ""
            for j in range(minLen):
                if w1[j] != w2[j]:
                    adj[w1[j].add(w2[j])]
                    break

        def dfs(char):
            if char in visited: return visited[char]  # if retruned true means its been visited
            visited[char] = True
            for neiChar in adj[char]:
                if dfs(neiChar): return True
            visited[char] = False
            res.append(char)

        for char in adj:
            if dfs(char): return ""
        res.reverse()
        return "".join(res)

    def numberOfGoodPaths(self, vals: list[int], edges: list[list[int]]) -> int:
        def get_root(i):
            if i == par[i]: return i
            par[i] = get_root(par[i])
            return par[i]

        def connect(i, j):
            i, j = get_root(i), get_root(j)
            if i != j:
                if sz[i] < sz[j]: i, j = j, i
                par[j] = i
                sz[i] += sz[j]
                if cur[i] == cur[j]:
                    r = cnt[i] * cnt[j]
                    cnt[i] += cnt[j]
                    return r
                elif cur[i] < cur[j]:
                    cur[i], cnt[i] = cur[j], cnt[j]
            return 0

        n = ans = len(vals)
        sz, cur, cnt, par = [1] * n, vals, [1] * n, list(range(n))
        for a, b in sorted(edges, key=lambda p: max(vals[p[0]], vals[p[1]])): ans += connect(a, b)
        return ans

    def maxNumEdgesToRemove(self, n: int, edges: list[list[int]]) -> int:
        edges, alex, bob, res = sorted(edges, key=lambda x: -x[0]), UnionFind(n), UnionFind(n), 0
        for traversal_type, source, destination in edges:
            if traversal_type == 3:
                if alex.union_cycle(source - 1, destination - 1): res += 1
                bob.union_cycle(source - 1, destination - 1)
            elif traversal_type == 1:
                if alex.union_cycle(source - 1, destination - 1): res += 1
            elif traversal_type == 2:
                if bob.union_cycle(source - 1, destination - 1): res += 1
        for i in range(n):
            alex.find(i)
            bob.find(i)
        total_alex_parents, total_bob_parents = len(set(alex.parents)), len(set(bob.parents))
        if total_alex_parents > 1 or total_bob_parents > 1: return -1
        return res

    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: list[list[int]]) -> list[list[int]]:
        for i, e in enumerate(edges): e.append(i)  # [v1, v2, weight, original_index]
        edges.sort(key=lambda e: e[2])
        mst_weight, uf, critical, pseudo = 0, UnionFind(n), [], []
        for v1, v2, w, i in edges:
            if uf.union(v1, v2): mst_weight += w
        for n1, n2, e_weight, i in edges:  # Try without curr edge
            weight, uf = 0, UnionFind(n)
            for v1, v2, w, j in edges:
                if i != j and uf.union(v1, v2): weight += w
            if max(uf.rank) != n or weight > mst_weight:
                critical.append(i)
                continue
            uf, weight = UnionFind(n), e_weight  # Try with curr edge
            uf.union(n1, n2)
            for v1, v2, w, j in edges:
                if uf.union(v1, v2): weight += w
            if weight == mst_weight: pseudo.append(i)
        return [critical, pseudo]

    #################### 1-D Dynamic Programming #################### 7/12 - 3

    def climbStairs(self, n: int) -> int:
        # if n == 0 or n == 1: return 1 # <- recursion
        # return self.climbStairs(n-1) + self.climbStairs(n-2)
        if n == 0 or n == 1: return 1
        prev, curr = 1, 1
        for i in range(2, n + 1):
            temp, curr = curr, prev + curr
            prev = temp
        return curr

    def minCostClimbingStairs(self, cost: list[int]) -> int:
        # for i in range(len(cost)-3, -1, -1): cost[i] += min(cost[i+1], cost[i+2])
        # return min(cost[0], cost[1]) # <- shorter but as efficient
        if not cost: return 0
        dp = [0] * len(cost)
        dp[0] = cost[0]
        if len(cost) >= 2: dp[1] = cost[1]
        for i in range(2, len(cost)): dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
        return min(dp[-1], dp[-2])

    def rob(self, nums: list[int]) -> int:
        rob1, rob2 = 0, nums[0]
        for i in range(1, len(nums)): rob1, rob2 = rob2, max(rob1 + nums[i], rob2)
        return rob2

    def robII(self, nums: list[int]) -> int:
        def help(nums):
            if len(nums) == 0: return 0
            if len(nums) == 1: return nums[0]
            if len(nums) == 2: return max(nums[0], nums[1])
            rob1, rob2 = 0, nums[0]
            for i in range(1, len(nums)): rob1, rob2 = rob2, max(rob1 + nums[i], rob2)
            return rob2

        return max(nums[0], help(nums[1:]), help(nums[:-1]))

    def longestPalindrome(self, s: str) -> str:
        res, resl = "", 0

        def getPal(l, r, res, resl):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resl: res, resl = s[l: r + 1], r - l + 1
                l, r = l - 1, r + 1
            return [res, resl]

        for i in range(len(s)):
            res, resl = getPal(i, i, res, resl)
            res, resl = getPal(i, i + 1, res, resl)
        return res

    def countSubstrings(self, s: str) -> int:
        res = 0

        def countPali(l, r):
            res = 0
            while l >= 0 and r < len(s) and s[l] == s[r]: res, l, r = res + 1, l - 1, r + 1
            return res

        for i in range(len(s)):
            res += countPali(i, i)
            res += countPali(i, i + 1)
        return res

    def lengthOfLIS(self, nums: list[int]) -> int:
        # dp = [1] * len(nums) # <- n^2
        # for i in range(len(nums) -1, -1, -1):
        #     for j in range(i+1, len(nums)):
        #         if nums[i] < nums[j]: dp[i] = max(dp[i], 1+dp[j])
        # return max(dp)
        dp, res = [0] * len(nums), 0  # <- binarysearch nlogn
        for num in nums:
            l, r = 0, res
            while l != r:
                mid = l + (r - l) // 2
                if dp[mid] < num:
                    l = mid + 1
                else:
                    r = mid
            res = max(res, l + 1)
            dp[l] = num
        return res

    """
    322. Coin Change
    You are given an integer array coins representing coins of different denominations 
    and an integer amount representing a total amount of money.
    Return the fewest number of coins that you need to make up that amount. 
    If that amount of money cannot be made up by any combination of the coins, return -1.
    You may assume that you have an infinite number of each kind of coin.
    Ex1: Input: coins = [1,2,5], amount = 11, Output: 3
    Explanation: 11 = 5 + 5 + 1
    Ex2: Input: coins = [2], amount = 3, Output: -1
    Ex3: Input: coins = [1], amount = 0, Output: 0
    DP - Bottom-up (recurrence relation) - O(amount * len(coins), O(amount)
    """

    def coinChange(self, coins: list[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for j in coins:
                if i - j >= 0: dp[i] = min(dp[i], 1 + dp[i - j])
        return dp[amount] if dp[amount] != amount + 1 else -1

    """
    152. Maximum Product Subarray
    Given an integer array nums, find a subarray that has the largest product, and return the product.
    The test cases are generated so that the answer will fit in a 32-bit integer.
    Ex1: Input: nums = [2,3,-2,4], Output: 6
    Explanation: [2,3] has the largest product 6.
    Ex2: Input: nums = [-2,0,-1], Output: 0
    Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
    """

    def maxProduct(self, nums: list[int]) -> int:
        res, curMin, curMax = nums[0], 1, 1
        for num in nums:
            tempMax, tempMin = curMax * num, curMin * num
            curMax, curMin = max(tempMax, tempMin, num), min(tempMax, tempMin, num)
            res = max(res, curMax)
        return res

    """
    139. Word Break
    Given a string s and a dictionary of strings wordDict, 
    return true if s can be segmented into a space-separated sequence 
    of one or more dictionary words.
    Note that the same word in the dictionary may be reused multiple times in the segmentation.
    Ex1: Input: s = "leetcode", wordDict = ["leet","code"], Output: true
    Explanation: Return true because "leetcode" can be segmented as "leet code".
    Ex2: Input: s = "applepenapple", wordDict = ["apple","pen"], Output: true
    Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
    Note that you are allowed to reuse a dictionary word.
    Ex3: Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"], Output: false
    """

    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True
        for i in range(len(s) - 1, -1, -1):
            for w in wordDict:
                if (i + len(w)) <= len(s) and s[i: i + len(w)] == w:
                    dp[i] = dp[i + len(w)]
                if dp[i]: break
        return dp[0]

    """
    416. Partition Equal Subset Sum
    Given an integer array nums, return true if you can partition the array into two subsets 
    such that the sum of the elements in both subsets is equal or false otherwise.
    Ex1: Input: nums = [1,5,11,5], Output: true
    Explanation: The array can be partitioned as [1, 5, 5] and [11].
    Ex2: Input: nums = [1,2,3,5], Output: false
    Explanation: The array cannot be partitioned into equal sum subsets.
    """

    def canpartition(self, nums: list[int]) -> bool:
        if sum(nums) % 2: return False
        dp, target = set(), sum(nums) // 2
        dp.add(0)
        for i in range(len(nums) - 1, -1, -1):
            nextDp = set()
            for t in dp:
                if (t + nums[i]) == target: return True
                nextDp.add(t + nums[i])
                nextDp.add(t)
            dp = nextDp
        return False

    """
    91. Decode Ways
    A message containing letters from A-Z can be encoded into numbers using the following mapping:
    'A' -> "1"
    'B' -> "2"
    ...
    'Z' -> "26"
    To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of 
    the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
    "AAJF" with the grouping (1 1 10 6), "KJF" with the grouping (11 10 6)
    Note that the grouping (1 11 06) is invalid because "06" cannot be mapped 
    into 'F' since "6" is different from "06".
    Given a string s containing only digits, return the number of ways to decode it.
    The test cases are generated so that the answer fits in a 32-bit integer.
    Ex1: Input: s = "12", Output: 2
    Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
    Ex2: Input: s = "226", Output: 3
    Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
    Ex3: Input: s = "06", Output: 0
    Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
    """

    def numDecodings(self, s: str) -> int:
        # Memoization - 43ms, 16mb
        dp = {len(s): 1}
        # def dfs(i):
        #     if i in dp: return dp[i]
        #     if s[i] == "0": return 0
        #     res = dfs(i + 1)
        #     if i + 1 < len(s) and (s[i] == "1" or s[i] == "2" and s[i+1] in "0123456"):
        #         res += dfs(i+2)
        #     dp[i] = res
        #     return res
        # return dfs(0)

        # Dynamic programming - 34ms, 16.2mb
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]
            if i + 1 < len(s) and (s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"):
                dp[i] += dp[i + 2]
        return dp[0]

    # new Picks from 300 list

    def minimumTotal(self, triangle: list[list[int]]) -> int:
        dp = triangle[-1]
        for r in range(len(triangle) - 2, -1, -1):
            for c in range(0, r + 1):
                dp[c] = triangle[r][c] + min(dp[c], dp[c + 1])
        return dp[0]

    def deleteAndEarn(self, nums: list[int]) -> int:
        store, dp = [0] * (max(nums) + 1), [0] * (max(nums) + 1)
        for num in nums: store[num] += num
        dp[1] = 1 * store[1]
        for i in range(2, max(nums) + 1): dp[i] = max(dp[i - 2] + store[i], dp[i - 1])
        return dp[-1]

    def combinationSum4(self, nums: list[int], target: int) -> int:
        dp = [0] * (target + 1)
        for i in range(1, target + 1):
            for num in nums:
                if num == i: dp[i] += 1
                if num < i: dp[i] += dp[i - num]
        return dp[-1]

    #################### 2-D Dynamic Programming #################### /11 (4 HARD) -

    def uniquePath(self, m: int, n: int) -> int:
        row = [1] * n
        for i in range(m - 1):
            newRow = [1] * n
            for j in range(n - 2, -1, -1): newRow[j] = newRow[j + 1] + row[j]
            row = newRow
        return row[0]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) - 1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[0][0]

    def maxProfitII(self, prices: list[int]) -> int:
        # buy, sell, cooldown = float('-inf'), 0, 0 # <- less code less memory but not as optimized
        # for i in range(len(prices)):
        #     buy, cooldown = max(buy, cooldown - prices[i]), sell
        #     sell = max(sell, buy + prices[i])
        # return sell
        # If Buy -> i + 1
        # If Sell -> i + 2
        dp = {}  # key=(i, buying) val=max_profit

        def dfs(i, buying):
            if i >= len(prices): return 0
            if (i, buying) in dp: return dp[(i, buying)]
            cooldown = dfs(i + 1, buying)
            if buying:
                buy = dfs(i + 1, not buying) - prices[i]
                dp[(i, buying)] = max(buy, cooldown)
            else:
                sell = dfs(i + 2, not buying) + prices[i]
                dp[(i, buying)] = max(sell, cooldown)
            return dp[(i, buying)]

        return dfs(0, True)

    def change(self, amount: int, coins: list[int]) -> int:
        # dp = {}
        # def dfs(i, a):
        #     if a == amount: return 1
        #     if a > amount or i == len(coins): return 0
        #     if (i, a) in dp: return dp[(i, a)]
        #     dp[(i, a)] = dfs(i, a + coins[i]) + dfs(i+1, a)
        #     return dp[(i, a)]
        # return dfs(0, 0)
        dp = [0] * (amount + 1)  # <- more optimized 1D solution
        dp[0] = 1
        for c in coins:
            for a in range(c, amount + 1): dp[a] += dp[a - c]
        return dp[amount]

    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        # dp = {}
        # def dfs(i, total):
        #     if i == len(nums): return 1 if total == target else 0
        #     if (i, total) in dp: return dp[(i, total)]
        #     dp[(i, total)] = dfs(i+1, total + nums[i]) + dfs(i+1, total - nums[i])
        #     return dp[(i, total)]
        # return dfs(0, 0)
        total = sum(nums)  # <- more optimized
        if total < abs(target) or (total + target) & 1: return 0

        def knapsack(target):
            dp = [1] + [0] * total
            for num in nums:
                for j in range(total, num - 1, -1): dp[j] += dp[j - num]
            return dp[target]

        return knapsack((total + target) // 2)

    """
    97. Interleaving String
    Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
    An interleaving of two strings s and t is a configuration where s and t are divided into n and m 
    substrings respectively, such that:
        s = s1 + s2 + ... + sn
        t = t1 + t2 + ... + tm
        |n - m| <= 1
    The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
    Note: a + b is the concatenation of strings a and b.
    Ex1: Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac", Output: true
    Explanation: One way to obtain s3 is:
        Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
        Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
        Since s3 can be obtained by interleaving s1 and s2, we return true.
    Ex2: Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc", Output: false
    Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.
    Ex3: Input: s1 = "", s2 = "", s3 = "", Output: true
    """

    def isInterLeave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3): return False
        dp = [[False] * (len(s2) + 1) for i in range(len(s1) + 1)]
        dp[len(s1)][len(s2)] = True
        for i in range(len(s1), -1, -1):
            for j in range(len(s2), -1, -1):
                if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]: dp[i][j] = True
                if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]: dp[i][j] = True
        return dp[0][0]

    """
    329. Longest Increasing Path in a Matrix - HARD
    Given an m x n integers matrix, return the length of the longest increasing path in matrix.
    From each cell, you can either move in four directions: left, right, up, or down. 
    You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).
    Ex1: Input: matrix = [[9,9,4],[6,6,8],[2,1,1]], Output: 4
    Explanation: The longest increasing path is [1, 2, 6, 9].
    Ex2: Input: matrix = [[3,4,5],[3,2,6],[2,2,1]], Output: 4
    Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
    """

    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        ROWS, COLS, dp = len(matrix), len(matrix[0]), {}  # (r, c) -> LIP

        def dfs(r, c, preVal):
            if r < 0 or r == ROWS or c < 0 or c == COLS or matrix[r][c] <= preVal: return 0
            if (r, c) in dp: return dp[(r, c)]
            res = 1
            res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
            res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))
            dp[(r, c)] = res
            return res

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, -1)
        return max(dp.values())

    """
    115. Distinct Subsequences - HARD
    Given two strings s and t, return the number of distinct subsequences of s which equals t.
    The test cases are generated so that the answer fits on a 32-bit signed integer.
    Ex1: Input: s = "rabbbit", t = "rabbit", Output: 3
    Explanation:
        As shown below, there are 3 ways you can generate "rabbit" from s.
        'rab'bb'it'
        'ra'b'bbit'
        'rab'b'bit'
    Ex2: Input: s = "babgbag", t = "bag", Output: 5
    Explanation:
    As shown below, there are 5 ways you can generate "bag" from s.
        'ba'b'g'bag
        'ba'bgba'g'
        'b'abgb'ag'
        ba'b'gb'ag'
        babg'bag'
    """

    def numDistinct(self, s: str, t: str) -> int:
        dp = {}
        for i in range(len(s) + 1): dp[(i, len(t))] = 1
        for j in range(len(t)): dp[(len(s), j)] = 0
        for i in range(len(s) - 1, -1, -1):
            for j in range(len(t) - 1, -1, -1):
                if s[i] == t[j]:
                    dp[(i, j)] = dp[(i + 1, j + 1)] + dp[(i + 1, j)]
                else:
                    dp[(i, j)] = dp[(i + 1, j)]
        return dp[(0, 0)]

    """
    72. Edit Distance
    Given two strings word1 and word2, return the minimum number of 
    operations required to convert word1 to word2.
    You have the following three operations permitted on a word:
    Insert a character, Delete a character, Replace a character
    Ex1: Input: word1 = "horse", word2 = "ros", Output: 3
    Explanation: 
        horse -> rorse (replace 'h' with 'r')
        rorse -> rose (remove 'r')
        rose -> ros (remove 'e')
    Ex2: Input: word1 = "intention", word2 = "execution", Output: 5
    Explanation: 
        intention -> inention (remove 't')
        inention -> enention (replace 'i' with 'e')
        enention -> exention (replace 'n' with 'x')
        exention -> exection (replace 'n' with 'c')
        exection -> execution (insert 'u')
    """

    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[float('inf')] * (len(word2) + 1) for i in range(len(word1) + 1)]
        for j in range(len(word2) + 1): dp[len(word1)][j] = len(word2) - j
        for i in range(len(word1) + 1): dp[i][len(word2)] = len(word1) - i
        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2) - 1, -1, -1):
                if word1[i] == word2[j]:
                    dp[i][j] = dp[i + 1][j + 1]
                else:
                    dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
        return dp[0][0]

    """
    312. Burst Balloons - HARD
    You are given n balloons, indexed from 0 to n - 1. Each balloon is painted 
    with a number on it represented by an array nums. You are asked to burst all the balloons.
    If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. If i - 1 or i + 1 
    goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.
    Return the maximum coins you can collect by bursting the balloons wisely.
    Ex1: Input: nums = [3,1,5,8], Output: 167
    Explanation:
        nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
        coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
    Ex2: Input: nums = [1,5], Output: 10
    """

    def maxCoins(self, nums: list[int]) -> int:
        dp, nums = {}, [1] + nums + [1]
        for offset in range(2, len(nums)):
            for left in range(len(nums) - offset):
                right = left + offset
                for pivot in range(left + 1, right):
                    coins = nums[left] * nums[pivot] * nums[right]
                    coins += dp.get((left, pivot), 0) + dp.get((pivot, right), 0)
                    dp[(left, right)] = max(coins, dp.get((left, right), 0))
        return dp.get((0, len(nums) - 1), 0)

    """
    10. Regular Expression Matching - HARD
    Given an input string s and a pattern p, 
    implement regular expression matching with support for '.' and '*' where:
    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.
    The matching should cover the entire input string (not partial).
    Ex1: Input: s = "aa", p = "a", Output: false
    Explanation: "a" does not match the entire string "aa".
    Ex2: Input: s = "aa", p = "a*", Output: true
    Explanation: '*' means zero or more of the preceding element, 'a'. 
        Therefore, by repeating 'a' once, it becomes "aa".
    Ex3: Input: s = "ab", p = ".*", Output: true
    Explanation: ".*" means "zero or more (*) of any character (.)".
    BOTTOM-UP Dynamic Programming
    """

    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False] * (len(p) + 1) for i in range(len(s) + 1)]
        dp[len(s)][len(p)] = True
        for i in range(len(s), -1, -1):
            for j in range(len(p) - 1, -1, -1):
                match = i < len(s) and (s[i] == p[j] or p[j] == ".")
                if (j + 1) < len(p) and p[j + 1] == "*":
                    dp[i][j] = dp[i][j + 2]
                    if match: dp[i][j] = dp[i + 1][j] or dp[i][j]
                elif match:
                    dp[i][j] = dp[i + 1][j + 1]
        return dp[0][0]

    # new Picks from 300 list

    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        if not obstacleGrid or obstacleGrid[0][0] == 1: return 0
        ROWS, COLS, dp = len(obstacleGrid), len(obstacleGrid[0]), [0] * len(obstacleGrid[0])
        dp[0] = 1
        for r in range(ROWS):
            for c in range(COLS):
                if obstacleGrid[r][c] == 1:
                    dp[c] = 0
                else:
                    if c > 0: dp[c] += dp[c - 1]
        return dp[COLS - 1]

    def longestPalindromeSubseq(self, s: str) -> int:
        def lcs(s1, s2):
            N, M = len(s1), len(s2)
            dp = [[0] * (M + 1) for _ in range(N + 1)]
            for i in range(N):
                for j in range(M):
                    if s1[i] == s2[j]:
                        dp[i + 1][j + 1] = 1 + dp[i][j]
                    else:
                        dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
            return dp[N][M]

        return lcs(s, s[::-1])

    def lastStoneWeightII(self, stones: list[int]) -> int:
        return min(functools.reduce(lambda dp, y: {x + y for x in dp} | {abs(x - y) for x in dp}, stones, {0}))

    def minPathSum(self, grid: list[list[int]]) -> int:
        M, N, prev = len(grid), len(grid[0]), [float('inf')] * len(grid[0])
        prev[-1] = 0
        for row in range(M - 1, -1, -1):
            dp = [float('inf')] * N
            for col in range(N - 1, -1, -1):
                if col < N - 1: dp[col] = min(dp[col], dp[col + 1])
                dp[col] = min(dp[col], prev[col]) + grid[row][col]
            prev = dp
        return prev[0]

    def maximalSquare(self, matrix: list[list[str]]) -> int:
        res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "0":
                    matrix[i][j] = 0
                else:
                    if i == 0 or j == 0:
                        matrix[i][j] = 1
                    else:
                        matrix[i][j] = min(matrix[i][j - 1], matrix[i - 1][j], matrix[i - 1][j - 1]) + 1
                res = max(res, matrix[i][j])
        return res ** 2

    #################### Greedy ####################### /8 -

    def maxSubArray(self, nums: list[int]) -> int:
        res, total = nums[0], 0
        for num in nums:
            total += num
            res = max(res, total)
            if total < 0: total = 0
        return res

    def canJump(self, nums: list[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= goal: goal = i
        return goal == 0

    def jump(self, nums: list[int]) -> int:
        l, r, res = 0, 0, 0
        while r < len(nums) - 1:
            maxJump = 0
            for i in range(l, r + 1): maxJump = max(maxJump, i + nums[i])
            l, r = r + 1, maxJump
            res += 1
        return res

    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        totalGas, remain, ind = 0, 0, 0
        for i in range(len(gas)):
            totalGas += gas[i] - cost[i]
            remain += gas[i] - cost[i]
            if remain < 0: remain, ind = 0, i + 1
        return ind if totalGas >= 0 else -1

    def isNStraightHand(self, hand: list[int], groupSize: int) -> bool:
        if len(hand) % groupSize: return False
        count = Counter(hand)
        minH = list(count.keys())
        heapq.heapify(minH)
        while minH:
            first = minH[0]
            for i in range(first, first + groupSize):
                if i not in count: return False
                count[i] -= 1
                if count[i] == 0:
                    if i != minH[0]: return False
                    heapq.heappop(minH)
        return True

    def mergeTriplets(self, triplets: list[list[int]], target: list[int]) -> bool:
        # return all(any(target[i] == t[i] and t[i-1] <= target[i-1] and t[i-2] <= target[i-2] for t in triplets) for i in range(3))
        good = set()
        for t in triplets:
            if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]: continue
            for i, v in enumerate(t):
                if v == target[i]: good.add(i)
        return len(good) == 3

    def partitionLabels(self, s: str) -> list[int]:
        lastInd, res, start, end = {c: i for (i, c) in enumerate(s)}, [], 0, 0
        for i, c in enumerate(s):
            end = max(end, lastInd[c])
            if i == end:
                res.append(end - start + 1)
                start = end + 1
        return res

    def checkVaildString(self, s: str) -> bool:
        leftMin, leftMax = 0, 0
        for c in s:
            if c == "(":
                leftMin, leftMax = leftMin + 1, leftMax + 1
            elif c == ")":
                leftMin, leftMax = leftMin - 1, leftMax - 1
            else:
                leftMin, leftMax = leftMin - 1, leftMax + 1
            if leftMax < 0: return False
            if leftMin < 0: leftMin = 0  # required because -> s = ( * ) (
        return leftMin == 0

    def maxSubarraySumCircular(self, nums: list[int]) -> int:
        globMax, globMin, curMax, curMin, total = nums[0], nums[0], 0, 0, 0
        for num in nums:
            curMax, curMin = max(curMax + num, num), min(curMin + num, num)
            globMax, globMin = max(curMax, globMax), min(curMin, globMin)
            total += num
        return max(globMax, total - globMin) if globMax > 0 else globMax

    def maxTurbulenceSize(self, arr: list[int]) -> int:
        down = up = res = 1
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                down = up + 1
                up = 1
            elif arr[i] < arr[i - 1]:
                up = down + 1
                down = 1
            else:
                down = 1
                up = 1
            res = max(res, up, down)
        return res

    #################### Intervals #################### /6 (1 HARD) -

    def canAttend(self, intervals: list[list[int]]) -> bool:
        intervals.sort(key=lambda i: i[0])
        for i in range(1, len(intervals)):
            i1, i2 = intervals[i - 1], intervals[i]
            if i1[1] > i2[0]: return False
        return True

    def minMeetingRooms(self, intervals: list[list[int]]) -> int:
        time, count, maxCount = [], 0, 0
        for start, end in intervals:
            time.append((start, 1))
            time.append((end, -1))
        time.sort(key=lambda x: (x[0], x[1]))
        for t in time:
            count += t[1]
            maxCount = max(maxCount, count)
        return maxCount

    def mergeIntervals(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort(key=lambda pair: pair[0])
        res = [intervals[0]]
        for start, end in intervals:
            if res[-1][1] < start:
                res.append([start, end])
            else:
                res[-1][1] = max(res[-1][1], end)  # merge
        return res

    def insertInterval(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
        res = []
        for i, (start, end) in enumerate(intervals):
            if start > newInterval[1]:
                res.append(newInterval)
                return res + intervals[i:]
            elif end < newInterval[0]:
                res.append(intervals[i])
            else:
                newInterval = [min(start, newInterval[0]), max(end, newInterval[1])]
        res.append(newInterval)
        return res

    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        # intervals.sort()
        # res, prevEnd = 0, intervals[0][1]
        # for start, end in intervals[1:]:
        #     if start >= prevEnd: prevEnd = end
        #     else: res, prevEnd = res+1, min(end, prevEnd)
        # return res
        intervals.sort(key=lambda x: x[1])  # <- much faster
        prev, count = 0, 1
        for i in range(1, len(intervals)):
            if intervals[i][0] >= intervals[prev][1]: prev, count = i, count + 1
        return len(intervals) - count

    def minInterval(self, intervals: list[list[int]], queries: list[int]) -> list[int]:
        intervals.sort()
        minH, res, i = [], {}, 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(minH, (r - l + 1, r))
                i += 1
            while minH and minH[0][1] < q: heapq.heappop(minH)
            res[q] = minH[0][0] if minH else -1
        return [res[q] for q in queries]

    # new Picks from 300 list - last 2

    def removeCoveredIntervals(self, intervals: list[list[int]]) -> int:
        res, long = len(intervals), 0
        intervals = sorted(intervals, key=lambda i: (i[0], -i[1]))
        for _, end in intervals:
            if end <= long:
                res -= 1
            else:
                long = end
        return res

    """
    352. Data Stream as Disjoint Intervals
    SummaryRange CLASS - HARD
    """

    #################### Math & Geometry #################### /8 -

    def isHappy(self, n: int) -> bool:
        res = set()
        while n != 1:
            if n in res: return False
            res.add(n)
            n = sum([int(i) ** 2 for i in str(n)])
        return True

    def plusOne(self, digits: list[int]) -> list[int]:
        # one, i, digits = 1, 0, digits[::-1]
        # while one:
        #     if i < len(digits):
        #         if digits[i] == 9: digits[i] = 0
        #         else:
        #             digits[i] += 1
        #             one = 0
        #     else:
        #         digits.append(one)
        #         one = 0
        #     i += 1
        # return digits[::-1]
        res = ""  # <- much faster
        for num in digits: res += str(num)
        temp = str(int(res) + 1)
        return [int(temp[i]) for i in range(len(temp))]

    def rotate(self, matrix: list[list[int]]) -> None:
        # l, r = 0, len(matrix) - 1
        # while l < r:
        #     for i in range(r - l):
        #         t, b = l, r
        #         topLeft        = matrix[t][l+i]      # save the topLeft
        #         matrix[t][l+i] = matrix[b-i][l]      # move bottom left into top left
        #         matrix[b-i][l] = matrix[b][r-i]      # move bottom right into bottom left
        #         matrix[b][r-i] = matrix[t+i][r]      # move top right into bottom right
        #         matrix[t+i][r] = topLeft             # move top left into top right
        #     r -= 1
        #     l += 1
        for i in range(len(matrix)):  # <- less efficient but less code
            for j in range(len(matrix) - 1, -1, -1):
                matrix[i].append(matrix[j].pop(0))

    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        res, left, right, top, bottom = [], 0, len(matrix[0]), 0, len(matrix)
        while left < right and top < bottom:
            for i in range(left, right): res.append(matrix[top][i])  # get top row
            top += 1
            for i in range(top, bottom): res.append(matrix[i][right - 1])  # get right col
            right -= 1
            if not (left < right and top < bottom): break
            for i in range(right - 1, left - 1, -1): res.append(matrix[bottom - 1][i])  # get bottom row
            bottom -= 1
            for i in range(bottom - 1, top - 1, -1): res.append(matrix[i][left])  # get left col
            left += 1
        return res

    def setZeroes(self, matrix: list[list[int]]) -> None:
        # ROWS, COLS, rowZero = len(matrix), len(matrix[0]), 0
        # for r in range(ROWS):
        #     for c in range(COLS):
        #         if matrix[r][c] == 0:
        #             matrix[0][c] = 0
        #             if r > 0: matrix[r][0] = 0
        #             else: rowZero = True
        # for r in range(1, ROWS):
        #     for c in range(1, COLS):
        #         if matrix[0][c] == 0 or matrix[r][0] == 0: matrix[r][c] = 0
        # if matrix[0][0] == 0:
        #     for r in range(ROWS): matrix[r][0] = 0
        # if rowZero:
        #     for c in range(COLS): matrix[0][c] = 0
        ROWS, COLS = len(matrix), len(matrix[0])  # <- faster and shorter
        res = [[i, j] for i in range(ROWS) for j in range(COLS) if matrix[i][j] == 0]
        for k, l in res:
            for row in range(COLS): matrix[k][row] = 0
            for col in range(ROWS): matrix[col][l] = 0

    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if x == 0: return 0
            if n == 0: return 1
            res = helper(x * x, n // 2)
            return x * res if n % 2 else res

        res = helper(x, abs(n))
        return res if n >= 0 else 1 / res

    def multiply(self, num1: str, num2: str) -> str:
        # if "0" in [num1, num2]: return "0"
        # res = [0] * (len(num1) + len(num2))
        # num1, num2 = num1[::-1], num2[::-1]
        # for i1 in range(len(num1)):
        #     for i2 in range(len(num2)):
        #         digit, ind = int(num1[i1]) * int(num2[i2]), i1 + i2
        #         res[ind] += digit
        #         res[ind + 1] += res[ind] // 10
        #         res[ind] = res[ind] % 10
        # res, beg = res[::-1], 0
        # while beg < len(res) and res[beg] == 0: beg += 1
        # res = map(str, res[beg:])
        # return "".join(res)
        num = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        r1, r2 = 0, 0
        for i in num1: r1 = 10 * r1 + num[i]
        for i in num2: r2 = 10 * r2 + num[i]
        return str(r1 * r2)

    """
    2013. Detect Squares - DetectSquares CLASS
    """

    # new Picks from 300 list

    def numIsPanlindrome(self, x: int) -> bool:
        x = str(x)
        return x == x[::-1]

    def intToRoman(self, num: int) -> str:
        num_map, res = {
            1: "I",
            5: "V", 4: "IV",
            10: "X", 9: "IX",
            50: "L", 40: "XL",
            100: "C", 90: "XC",
            500: "D", 400: "CD",
            1000: "M", 900: "CM",
        }, ""
        for n in [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]:
            # If n in list then add the roman value to result variable
            while n <= num:
                res += num_map[n]
                num -= n
        return res

    #################### Bit Manipulation #################### 7/7 - 3

    def singleNumber(self, nums: list[int]) -> int:
        # res = 0
        # for n in nums: res = n ^ res
        # return res
        return functools.reduce(lambda x, y: x ^ y, nums, 0)

    def hammingWeight(self, n: int) -> int:
        return bin(n)[:].count('1')  # <- quick one-liner
        # res = 0
        # while n:
        #     n &= n - 1
        #     res += 1
        # return res

    def countBits(self, n: int) -> list[int]:
        # dp, offset = [0] * (n+1), 1
        # for i in range(1, n+1):
        #     if offset * 2 == i: offset = i
        #     dp[i] = 1 + dp[i - offset]
        # return dp
        return [bin(i)[:].count('1') for i in range(0, n + 1)]  # <- one-liner

    def reverseBits(self, n: int) -> int:
        # res = 0
        # for i in range(32):
        #     bit = (n >> i) & 1
        #     res += (bit << (31 - i)) # Ex: 01 << 1 = 10
        # return res
        def reverse(n, rev, bit):  # <- must faster
            return rev << (32 - bit) if n < 1 else reverse(n >> 1, rev << 1 | (n & 1), bit + 1)

        return reverse(n, 0, 0)

    def missingNumber(self, nums: list[int]) -> int:
        # res = len(nums)
        # for i in range(len(nums)): res += i - nums[i]
        # return res
        return (len(nums) * (len(nums) + 1)) // 2 - sum(nums)  # <- one-liner, faster

    def getSum(self, a: int, b: int) -> int:
        def add(a, b):
            if not a or not b: return a or b
            return add(a ^ b, (a & b) << 1)

        if a * b < 0:  # assume a < 0, b > 0
            if a > 0: return self.getSum(b, a)
            if add(~a, 1) == b: return 0  # -a == b
            if add(~a, 1) < b:  # -a < b
                return add(~add(add(~a, 1), add(~b, 1)), 1)  # -add(-a, -b)
        return add(a, b)  # a*b >= 0 or (-a) > b > 0

    def reverse(self, x: int) -> int:
        # MIN, MAX, res = -2147483648, 2147483647, 0  # -2^31, 2^31 - 1, 0
        # while x:
        #     digit = int(math.fmod(x, 10))    # (python dumb) -1 %  10 = 9
        #     x = int(x / 10)                     # (python dumb) -1 // 10 = -1
        #     if res > MAX // 10 or (res == MAX // 10 and digit > MAX % 10): return 0
        #     if res < MIN // 10 or (res == MIN // 10 and digit < MIN % 10): return 0
        #     res = (res * 10) + digit
        # return res
        if x >= 0:
            y = int(str(x)[::-1])
            return y if y < 2147483648 else 0
        else:
            y = -int(str(x)[:0:-1])
            return y if y > -2147483648 else 0

    def shuffle(self, nums: list[int], n: int) -> list[int]:
        # left, right, res = 0, n, []
        # while left < n:
        #     res.append(nums[left])
        #     res.append(nums[right])
        #     left, right = left + 1, right + 1
        # return res
        return [nums[i + (j % 2) * (len(nums) // 2)] for i in range(len(nums) // 2) for j in range(2)]

    def addToArrayForm(self, num: list[int], k: int) -> list[int]:
        for i in range(len(num) - 1, -1, -1): k, num[i] = divmod(num[i] + k, 10)
        while k:
            k, a = divmod(k, 10)
            num = [a] + num
        return num

    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a, 2) + int(b, 2))[2:]

    ######################## Daily Streaks ##########################

    """
    542. 01 Matrix
    Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
    The distance between two adjacent cells is 1.
    Ex1: Input: mat = [[0,0,0],[0,1,0],[1,1,1]], Output: [[0,0,0],[0,1,0],[1,2,1]]
    """
    def updateMatrix(self, mat: list[list[int]]) -> list[list[int]]:
        if not mat or len(mat) == 0 or len(mat[0]) == 0: return mat
        ROWS, COLS, visited = len(mat), len(mat[0]), []
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for r in range(ROWS):
            for c in range(COLS):
                if mat[r][c] == 0:
                    visited.append((r, c))
                else:
                    mat[r][c] = "#"

        for row, col in visited:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS and mat[r][c] == "#":
                    mat[r][c] = mat[row][col] + 1
                    visited.append((r, c))
        return mat

    """
    1203. Sort Items by Groups Respecting Dependencies
    """
    def sortItems(self, n: int, m: int, group: list[int], beforeItems: list[list[int]]) -> list[int]:
        group_item = [[] for _ in range(m + n)]
        inner_group_edges = [[] for _ in range(n)]
        inter_group_edges = [[] for _ in range(m + n)]
        inner_group_indeg = [0 for _ in range(n)]
        inter_group_indeg = [0 for _ in range(m + n)]

        # assign new group for each item in group -1
        group_ind = m
        for i in range(len(group)):
            if group[i] == -1:
                group[i] = group_ind
                group_ind += 1
            group_item[group[i]].append(i)

        for i in range(n):
            cur_group = group[i]
            for prev_item in beforeItems[i]:
                prev_group = group[prev_item]
                if cur_group == prev_group:
                    inner_group_edges[prev_item].append(i)
                    inner_group_indeg[i] += 1
                else:
                    inter_group_edges[prev_group].append(cur_group)
                    inter_group_indeg[cur_group] += 1

        def topological_sort(edges, indeg, nodes):
            queue = []
            for node in nodes:
                if indeg[node] == 0:
                    queue.append(node)
            ans = []
            while queue:
                cur_node = queue.pop(0)
                ans.append(cur_node)
                for new_node in edges[cur_node]:
                    indeg[new_node] -= 1
                    if indeg[new_node] == 0:
                        queue.append(new_node)
            if len(ans) == len(nodes):
                return ans
            else:
                return []

        group_sort = topological_sort(inter_group_edges, inter_group_indeg, list(range(m + n)))
        if len(group_sort) == 0:
            return []
        ans = []
        for cur_group in group_sort:
            cur_nodes = group_item[cur_group]
            if len(cur_nodes) == 0:
                continue
            cur_group_sort = topological_sort(inner_group_edges, inner_group_indeg, cur_nodes)
            if len(cur_group_sort) == 0:
                return []
            ans.extend(cur_group_sort)
        return ans

    """
    459. Repeated Substring Pattern
    Given a string s, check if it can be constructed by taking a 
    substring of it and appending multiple copies of the substring together.
    Ex1: Input: s = "abab", Output: true
    Explanation: It is the substring "ab" twice.
    Ex2: Input: s = "abcabcabcabc", Output: true
    Explanation: It is the substring "abc" four times or the substring "abcabc" twice.
    """
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s + s)[1:-1]

    """
    168. Excel Sheet Column Title
    Given an integer columnNumber, return its corresponding column title as it appears in an 
    Excel sheet.
    For example:
        A -> 1
        B -> 2
        C -> 3
        ...
        Z -> 26
        AA -> 27
        AB -> 28 
        ...
    Ex1: Input: columnNumber = 1, Output: "A"
    Ex2: Input: columnNumber = 28, Output: "AB"
    Ex3: Input: columnNumber = 701, Output: "ZY"
    """
    def convertToTitle(self, columnNumber: int) -> str:
        if columnNumber < 27: return chr(ord("A") + (columnNumber - 1) % 26)
        res = "";
        while columnNumber > 0:
            if columnNumber % 26 == 0:
                res += chr(ord("A") + 25)
                columnNumber -= 1
            else:
                res += chr(ord("A") + columnNumber % 26 - 1)
            columnNumber //= 26
        return res[::-1]

    """
    767. Reorganize String
    Given a string s, rearrange the characters of s so that any two adjacent characters 
    are not the same.
    Return any possible rearrangement of s or return "" if not possible.
    Ex1: Input: s = "aab", Output: "aba"
    Ex2: Input: s = "aaab", Output: ""
    """
    def reorganizeString(self, s: str) -> str:
        minH = [[-val, char] for char, val in Counter(s).items()]
        heapq.heapify(minH)
        res, prev = "", None
        while minH or prev:
            if not minH and prev: return ""
            freq, char = heapq.heappop(minH)
            res += char
            freq += 1
            if prev:
                heapq.heappush(minH, prev)
                prev = None
            if freq != 0: prev = [freq, char]
        return res

    """
    646. Maximum Length of Pair Chain
    You are given an array of n pairs pairs where pairs[i] = [lefti, righti] and lefti < righti.
    A pair p2 = [c, d] follows a pair p1 = [a, b] if b < c. A chain of pairs can be formed in this fashion.
    Return the length longest chain which can be formed.
    You do not need to use up all the given intervals. You can select pairs in any order.
    Ex1: Input: pairs = [[1,2],[2,3],[3,4]], Output: 2
    Explanation: The longest chain is [1,2] -> [3,4].
    Ex2: Input: pairs = [[1,2],[7,8],[4,5]], Output: 3
    Explanation: The longest chain is [1,2] -> [4,5] -> [7,8].
    """
    def findLongestChain(self, pairs: list[list[int]]) -> int:
        if not pairs or len(pairs) == 0: return 0
        pairs.sort(key=lambda x: x[1])
        cur, res = float("-inf"), 0
        for pair in pairs:
            if cur < pair[0]:
                cur = pair[1]
                res += 1
        return res

    """
    403. Frog Jump
    A frog is crossing a river. The river is divided into some number of units, and at each unit, 
    there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
    Given a list of stones' positions (in units) in sorted ascending order, determine if the frog can cross 
    the river by landing on the last stone. Initially, 
    the frog is on the first stone and assumes the first jump must be 1 unit.
    If the frog's last jump was k units, its next jump must be either k - 1, k, or k + 1 units. 
    The frog can only jump in the forward direction.
    Ex1: Input: stones = [0,1,3,5,6,8,12,17], Output: true
    Explanation: The frog can jump to the last stone by jumping 1 unit to the 2nd stone, then 2 units to the 3rd stone, 
    then 2 units to the 4th stone, then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the 8th stone.
    Ex2: Input: stones = [0,1,2,3,4,8,9,11], Output: false
    Explanation: There is no way to jump to the last stone as the gap between the 5th and 6th stone is too large.
    """
    def canCross(self, stones: list[int]) -> bool:
        if stones[1] != 1: return False
        dp = {}

        def dfs(i, k):
            if i == len(stones) - 1: return True
            if (i, k) in dp: return dp[(i, k)]
            res = False
            for j in range(i + 1, len(stones)):
                if stones[i] + k == stones[j]: res = res or dfs(j, k)
                if stones[i] + k + 1 == stones[j]: res = res or dfs(j, k + 1)
                if stones[i] + k - 1 == stones[j]: res = res or dfs(j, k - 1)
            dp[(i, k)] = res
            return res

        return dfs(1, 1)

    """
    2483. Minimum Penalty for a shop
    You are given the customer visit log of a shop represented by a 0-indexed string customers 
    consisting only of characters 'N' and 'Y':
        -if the ith character is 'Y', it means that customers come at the ith hour
            whereas 'N' indicates that no customers come at the ith hour.
        -If the shop closes at the jth hour (0 <= j <= n), the penalty is calculated as follows:
    For every hour when the shop is open and no customers come, the penalty increases by 1.
    For every hour when the shop is closed and customers come, the penalty increases by 1.
    Return the earliest hour at which the shop must be closed to incur a minimum penalty.
    Note that if a shop closes at the jth hour, it means the shop is closed at the hour j.
    Ex1: Input: customers = "YYNY", Output: 2
    Explanation: 
    - Closing the shop at the 0th hour incurs in 1+1+0+1 = 3 penalty.
    - Closing the shop at the 1st hour incurs in 0+1+0+1 = 2 penalty.
    - Closing the shop at the 2nd hour incurs in 0+0+0+1 = 1 penalty.
    - Closing the shop at the 3rd hour incurs in 0+0+1+1 = 2 penalty.
    - Closing the shop at the 4th hour incurs in 0+0+1+0 = 1 penalty.
        Closing the shop at 2nd or 4th hour gives a minimum penalty. Since 2 is earlier, 
        the optimal closing time is 2.
    Ex2: Input: customers = "NNNNN", Output: 0
        Explanation: It is best to close the shop at the 0th hour as no customers arrive.
    Ex3: Input: customers = "YYYY", Output: 4
        Explanation: It is best to close the shop at the 4th hour as customers arrive at each hour.
    """
    def bestClosingTime(self, customers: str) -> int:
        max_score, score, best = 0, 0, -2
        for i, c in enumerate(customers):
            score += 1 if c == 'Y' else -1
            if score > max_score:
                max_score, best = score, i
        return best + 1

    """
    2707. Extra Characters in a String
    You are given a 0-indexed string s and a dictionary of words dictionary. 
    You have to break s into one or more non-overlapping substrings such that each substring is present 
    in dictionary. There may be some extra characters in s which are not present in any of the substrings.
    Return the minimum number of extra characters left over if you break up s optimally.
    Ex1: Input: s = "leetscode", dictionary = ["leet","code","leetcode"], Output: 1
    Explanation: We can break s in two substrings: "leet" from index 0 to 3 and "code" from index 5 to 8. 
    There is only 1 unused character (at index 4), so we return 1.
    Ex2: Input: s = "sayhelloworld", dictionary = ["hello","world"], Output: 3
    Explanation: We can break s in two substrings: "hello" from index 3 to 7 and "world" from index 8 to 12. 
    The characters at indices 0, 1, 2 are not used in any substring and thus 
    are considered as extra characters. Hence, we return 3.
    """
    def minExtraChar(self, s: str, dictionary: list[str]) -> int:
        max_val, dicSet = len(s) + 1, set(dictionary)
        dp = [max_val] * (len(s) + 1)
        dp[0] = 0
        for i in range(1, len(s) + 1):
            dp[i] = dp[i - 1] + 1
            for j in range(1, i + 1):
                if s[i - j:i] in dicSet: dp[i] = min(dp[i], dp[i - j])
        return dp[-1]

    """
    1359. Count All Vaild Pickup and Delievery options
    Given n orders, each order consist in pickup and delivery services. 
    Count all valid pickup/delivery possible sequences such that delivery(i) is always after of pickup(i). 
    Since the answer may be too large, return it modulo 10^9 + 7.
    Ex1: Input: n = 1, Output: 1
    Explanation: Unique order (P1, D1), Delivery 1 always is after of Pickup 1.
    Ex2: Input: n = 2, Output: 6
    Explanation: All possible orders: 
    (P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1).
    This is an invalid order (P1,D2,P2,D1) because Pickup 2 is after of Delivery 2.
    Ex3: Input: n = 3, Output: 90
    """
    def countOrders(self, n: int) -> int:
        # return functools.reduce(lambda x, y: x * y % (10**9+7), (v for x in range(1, n+1) for v in (x, 2*x-1)), 1)
        res = 1  # <- a bit quicker
        for i in range(2, n + 1): res = (res * (2 * i - 1) * i) % (10 ** 9 + 7)
        return res

    """
    1647. Minimum Deletions to Make Character Frequencies Unique
    A string s is called good if there are no two different characters in s that have the same frequency.
    Given a string s, return the minimum number of characters you need to delete to make s good.
    The frequency of a character in a string is the number of times it appears in the string. 
    For example, in the string "aab", the frequency of 'a' is 2, while the frequency of 'b' is 1.
    Ex1: Input: s = "aab", Output: 0, Explanation: s is already good.
    Ex2: Input: s = "aaabbbcc", Output: 2, Explanation: You can delete two 'b's resulting in the good string "aaabcc".
            Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".
    Ex3: Input: s = "ceabaacb", Output: 2, Explanation: You can delete both 'c's resulting in the good string "eabaab".
    Note that we only care about characters that are still in the string at the end (i.e. frequency of 0 is ignored).
    """
    def minDeletions(self, s: str) -> int:
        delete, used_freq = 0, set()
        for freq in Counter(s).values():
            while freq > 0 and freq in used_freq: freq, delete = freq - 1, delete + 1
            used_freq.add(freq)
        return delete

    """
    1631. Path with Minimum Effort
    You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of 
    size rows x columns, where heights[row][col] represents the height of cell (row, col). 
    You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, 
    (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, 
    and you wish to find a route that requires the minimum effort. A route's effort is the 
    maximum absolute difference in heights between two consecutive cells of the route.
    Return the minimum effort required to travel from the top-left cell to the bottom-right cell.
    Ex1: Input: heights = [[1,2,2],[3,8,2],[5,3,5]], Output: 2
    Explanation: The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.
    This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.
    Ex2: Input: heights = [[1,2,3],[3,8,4],[5,3,5]], Output: 1
    Explanation: The route of [1,2,3,4,5] 
    has a maximum absolute difference of 1 in consecutive cells, which is better than route [1,3,5,3,5].
    Ex3: Input: heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]], Output: 0
    Explanation: This route does not require any effort.
    """
    def minimumEffortPath(self, heights: list[list[int]]) -> int:
        ROWS, COLS, directions = len(heights), len(heights[0]), [[0, 1], [0, -1], [1, 0], [-1, 0]]
        dist = [[math.inf for _ in range(COLS)] for _ in range(ROWS)]
        dist[0][0], minHeap = 0, [(0, 0, 0)]
        while minHeap:
            effort, row, col = heapq.heappop(minHeap)
            if row == ROWS - 1 and col == COLS - 1: return effort
            for dx, dy in directions:
                r, c = row + dx, col + dy
                if 0 <= r < ROWS and 0 <= c < COLS:
                    new_effort = max(effort, abs(heights[row][col] - heights[r][c]))
                    if new_effort < dist[r][c]:
                        dist[r][c] = new_effort
                        heapq.heappush(minHeap, (new_effort, r, c))

    """
    847. Shortest Path Visiting all Nodes
    You have an undirected, connected graph of n nodes labeled from 0 to n - 1. 
    You are given an array graph where graph[i] is a list of all the nodes connected with node i by an edge.
    Return the length of the shortest path that visits every node. 
    You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.
    Ex1: Input: graph = [[1,2,3],[0],[0],[0]], Output: 4, Explanation: One possible path is [1,0,2,0,3]
    Ex2: Input: graph = [[1],[0,2,4],[1,3,4],[2],[1,2]], Output: 4, Explanation: One possible path is [0,1,4,2,3]
    """
    def shortestPathLength(self, graph: list[list[int]]) -> int:
        q = deque([(1 << i, i, 0) for i in range(len(graph))])
        visited = set((1 << i, i) for i in range(len(graph)))
        while q:
            mask, node, dist = q.popleft()
            if mask == (1 << len(graph)) - 1: return dist
            for nei in graph[node]:
                newMask = mask | (1 << nei)
                if (newMask, nei) not in visited:
                    visited.add((newMask, nei))
                    q.append((newMask, nei, dist + 1))

    """
    1337. The K weakest Rows in a matrix
    You are given an m x n binary matrix mat of 1's (representing soldiers) and 0's (representing civilians). 
    The soldiers are positioned in front of the civilians. 
    That is, all the 1's will appear to the left of all the 0's in each row.
    A row i is weaker than a row j if one of the following is true:
    The number of soldiers in row i is less than the number of soldiers in row j.
    Both rows have the same number of soldiers and i < j.
    Return the indices of the k weakest rows in the matrix ordered from weakest to strongest.
    Ex1: Input: mat = [[1,1,0,0,0],[1,1,1,1,0], [1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], k = 3, Output: [2,0,3]
    Explanation: The number of soldiers in each row is: 
    - Row 0: 2 
    - Row 1: 4 
    - Row 2: 1 
    - Row 3: 2 
    - Row 4: 5 
    The rows ordered from weakest to strongest are [2,0,3,1,4].
    Ex2: Input: mat = [[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]], k = 2, Output: [0,2]
    Explanation: The number of soldiers in each row is: 
    - Row 0: 1 
    - Row 1: 4 
    - Row 2: 1 
    - Row 3: 1 
    The rows ordered from weakest to strongest are [0,2,3,1].
    """
    def kWeakestRows(self, mat: list[list[int]], k: int) -> list[int]:
        # rows = [(sum(row), i) for i, row in enumerate(mat)] # <- much faster
        # rows.sort(key=lambda x: (x[0], x[1]))
        # return [row[1] for row in rows[:k]]
        minH = []
        for i, row in enumerate(mat):
            heapq.heappush(minH, (-sum(row), -i))
            if len(minH) > k: heapq.heappop(minH)
        return [-i for _, i in sorted(minH, reverse=True)]

    """
    1048. Longest String Chain
    You are given an array of words where each word consists of lowercase English letters.
    wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA 
    without changing the order of the other characters to make it equal to wordB. For example, "abc" is a predecessor 
    of "abac", while "cba" is not a predecessor of "bcad". A word chain is a sequence of words 
    [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, 
    and so on. A single word is trivially a word chain with k == 1. 
    Return the length of the longest possible word chain with words chosen from the given list of words.
    Ex1: Input: words = ["a","b","ba","bca","bda","bdca"], Output: 4
        Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
    Ex2: Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"], Output: 5
        Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].
    Ex3: Input: words = ["abcd","dbqca"], Output: 1
        Explanation: The trivial word chain ["abcd"] is one of the longest word chains.
        ["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.
    """
    def longestStrChain(self, words: list[str]) -> int:
        # word_set, memo = set(words), {} # <- recursive but a bit slower
        # def dfs(word):
        #     if word not in word_set: return 0
        #     if word in memo: return memo[word]
        #     res = 1
        #     for i in range(len(word)):
        #         next = word[:i] + word[i+1:]
        #         res = max(res, dfs(next) + 1)
        #     memo[word] = res
        #     return res
        # return max(dfs(word) for word in words)
        if not words: return 0
        words.sort(key=len)
        dp, res = {}, 0
        for word in words:
            dp[word] = 1
            for i in range(len(word)):
                prev = word[:i] + word[i + 1:]
                if prev in dp: dp[word] = max(dp[word], dp[prev] + 1)
            res = max(res, dp[word])
        return res

    """
    799. Champagne Tower
    Now after pouring some non-negative integer cups of champagne, return how full the jth glass 
    in the ith row is (both i and j are 0-indexed.) We stack glasses in a pyramid, where the 
    first row has 1 glass, the second row has 2 glasses, and so on until the 100th row.  
    Each glass holds one cup of champagne. Then, some champagne is poured into the first glass 
    at the top.  When the topmost glass is full, any excess liquid poured will fall equally to 
    the glass immediately to the left and right of it. When those glasses become full, 
    any excess champagne will fall equally to the left and right of those glasses, and so on. 
    (A glass at the bottom row has its excess champagne fall on the floor.)
    For example, after one cup of champagne is poured, the top most glass is full.  
    After two cups of champagne are poured, the two glasses on the second row are half full.  
    After three cups of champagne are poured, those two cups become full - there are 3 full 
    glasses total now.  After four cups of champagne are poured, the third row has the middle 
    glass half full, and the two outside glasses are a quarter full, as pictured below.
    Ex1: Input: poured = 1, query_row = 1, query_glass = 1, Output: 0.00000
    Explanation: We poured 1 cup of champange to the top glass of the tower (which is indexed as (0, 0)). 
        There will be no excess liquid so all the glasses under the top glass will remain empty.
    Ex2: Input: poured = 2, query_row = 1, query_glass = 1, Output: 0.50000
    Explanation: We poured 2 cups of champange to the top glass of the tower (which is indexed as (0, 0)). 
        There is one cup of excess liquid. The glass indexed as (1, 0) and the glass indexed as (1, 1) 
        will share the excess liquid equally, and each will get half cup of champange.
    Ex3: Input: poured = 100000009, query_row = 33, query_glass = 17, Output: 1.00000
    """
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        tower = [[0] * (i + 1) for i in range(query_row + 1)]
        tower[0][0] = poured
        for row in range(query_row):
            for glass in range(len(tower[row])):
                excess = (tower[row][glass] - 1) / 2.0
                if excess > 0:
                    tower[row + 1][glass] += excess
                    tower[row + 1][glass + 1] += excess
        return min(1.0, tower[query_row][query_glass])

    """
    316. Remove Duplicate Letters
    Given a string s, remove duplicate letters so that every letter appears once and only once. 
    You must make sure your result is the smallest in lexicographical order among all possible results.
    Ex1: Input: s = "bcabc", Output: "abc"
    Ex2: Input: s = "cbacdcbc", Output: "acdb"
    """
    def removeDuplicateLetters(self, s: str) -> str:
        stack, seen, last_occ = [], set(), {c: i for i, c in enumerate(s)}
        for i, c in enumerate(s):
            if c not in seen:
                while stack and c < stack[-1] and i < last_occ[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)

    """
    389. Find the Difference
    You are given two strings s and t. String t is generated by random shuffling string s and 
    then add one more letter at a random position. Return the letter that was added to t.
    Ex1: Input: s = "abcd", t = "abcde", Output: "e"
        Explanation: 'e' is the letter that was added.
    Ex2: Input: s = "", t = "y", Output: "y"
    """
    def findTheDifference(self, s: str, t: str) -> str:
        res = 0
        for c in s + t: res ^= ord(c)
        return chr(res)

    """
    880. Decoded String at Index
    You are given an encoded string s. To decode the string to a tape, the encoded string is 
    read one character at a time and the following steps are taken:
    If the character read is a letter, that letter is written onto the tape.
    If the character read is a digit d, the entire current tape is repeatedly written d - 1 
    more times in total.
    Given an integer k, return the kth letter (1-indexed) in the decoded string.
    Ex1: Input: s = "leet2code3", k = 10, Output: "o"
        Explanation: The decoded string is "leetleetcodeleetleetcodeleetleetcode".
        The 10th letter in the string is "o".
    Ex2: Input: s = "ha22", k = 5, Output: "h"
        Explanation: The decoded string is "hahahaha".
        The 5th letter is "h".
    Ex3: Input: s = "a2345678999999999999999", k = 1, Output: "a"
        Explanation: The decoded string is "a" repeated 8301530446056247680 times.
        The 1st letter is "a".
    """
    def decodeAtIndex(self, s: str, k: int) -> str:
        res = [1]
        for x in s[1:]:
            if res[-1] >= k: break
            if x.isdigit():
                res.append(res[-1] * int(x))
            else:
                res.append(res[-1] + 1)
        for i in reversed(range(len(res))):
            k %= res[i]
            if not k and s[i].isalpha(): return s[i]

    """
    905. Sort Array By Parity
    Given an integer array nums, move all the even integers at the beginning of the array followed
    by all the odd integers. Return any array that satisfies this condition.
    Ex1: Input: nums = [3,1,2,4], Output: [2,4,3,1]
        Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.
    Ex2: Input: nums = [0], Output: [0]
    """
    def sortArrayByParity(self, nums: list[int]) -> list[int]:
        nums[:] = [i for i in nums if i % 2 == 0] + [j for j in nums if j % 2 != 0]
        return nums

    """
    896. Monotonic Array
    An array is monotonic if it is either monotone increasing or monotone decreasing.
    An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. 
    An array nums is monotone decreasing if for all i <= j, nums[i] >= nums[j].
    Given an integer array nums, return true if the given array is monotonic, or false otherwise.
    Ex1: Input: nums = [1,2,2,3], Output: true
    Ex2: Input: nums = [6,5,4,4], Output: true
    Ex3: Input: nums = [1,3,2], Output: false
    """
    def isMonotonic(self, nums: list[int]) -> bool:
        if len(nums) < 2: return True
        inc, dec, dic = True, True, 0
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                if dic == 0:
                    dic = 1
                elif dic == -1:
                    return False

            elif nums[i] < nums[i - 1]:
                if dic == 0:
                    dic = -1
                elif dic == 1:
                    return False
        return True

    """
    412. FizzBuzz
    Given an integer n, return a string array answer (1-indexed) where:
    answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
    answer[i] == "Fizz" if i is divisible by 3.
    answer[i] == "Buzz" if i is divisible by 5.
    answer[i] == i (as a string) if none of the above conditions are true.
    Ex1: Input: n = 3, Output: ["1","2","Fizz"]
    Ex2: Input: n = 5, Output: ["1","2","Fizz","4","Buzz"]
    Ex3: Input: n = 15, 
        Output: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
    """
    def fizzBuzz(self, n: int) -> list[str]:
        res = []
        for i in range(1, n + 1):
            if i % 15 == 0: res.append("FizzBuzz")
            elif i % 5 == 0: res.append("Buzz")
            elif i % 3 == 0: res.append("Fizz")
            else: res.append(str(i))
        return res

    """
    1342. Number of Steps to reduce a number to zero
    Given an integer num, return the number of steps to reduce it to zero.
    In one step, if the current number is even, you have to divide it by 2, 
    otherwise, you have to subtract 1 from it.
    Ex1: Input: num = 14, Output: 6, Explanation: 
        Step 1) 14 is even; divide by 2 and obtain 7. 
        Step 2) 7 is odd; subtract 1 and obtain 6.
        Step 3) 6 is even; divide by 2 and obtain 3. 
        Step 4) 3 is odd; subtract 1 and obtain 2. 
        Step 5) 2 is even; divide by 2 and obtain 1. 
        Step 6) 1 is odd; subtract 1 and obtain 0.
    Ex2: Input: num = 8, Output: 4, Explanation: 
        Step 1) 8 is even; divide by 2 and obtain 4. 
        Step 2) 4 is even; divide by 2 and obtain 2. 
        Step 3) 2 is even; divide by 2 and obtain 1. 
        Step 4) 1 is odd; subtract 1 and obtain 0.
    Ex3: Input: num = 123, Output: 12
    """
    def numberOfSteps(self, num: int) -> int:
        res = 0
        while num != 0:
            if num % 2 == 0: num /= 2
            else: num -= 1
            res += 1
        return res

    """
    876. Middle of a LinkedList
    Given the head of a singly linked list, return the 
    middle node of the linked list, if there 
    are two middle nodes, return the second middle node
    Ex1: Input: head = [1,2,3,4,5], output: [3,4,5]
    Explanation: The middle node of the list is node 3.
    Ex2: Input: head = [1,2,3,4,5,6], Output: [4,5,6]
    Explanation: Since the list has two middle nodes 
        with values 3 and 4, we return the second one.
    """
    def middleNode(self, head: [ListNode]) -> [ListNode]:
        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        return slow.next if fast else slow

    """
    383. Ransom Note
    Given two strings ransomNote and magaine, return true if ransomNote
    can be constructed by using the letters from magazine and false otherwise
    Each letter in magazine can only be used once in ransomNote
    Ex1: Input: ransomNote = "a", magazine = "b", Output: false
    Ex2: Input: ransomNote = "aa", magazine = "ab", Output: false
    Ex3: Input: ransomNote = "aa", magazine = "aab", Output: true
    """
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        if len(ransomNote) > len(magazine): return False
        rn, mag = Counter(ransomNote), Counter(magazine)
        return len(rn - mag) == 0

    """
    977. Squares of a Sorted Array
    Given an integer array nums sorted in non-decreasing order, 
    return an array of the squares of each number sorted in non-decreasing order.
    Ex1: Input: nums = [-4,-1,0,3,10], Output: [0,1,9,16,100]
        Explanation: After squaring, the array becomes [16,1,0,9,100].
        After sorting, it becomes [0,1,9,16,100].
    Ex2: Input: nums = [-7,-3,2,3,11], Output: [4,9,9,49,121]
    """
    def sortedSquares(self, nums: list[int]) -> list[int]:
        res, l, r = [0] * len(nums), 0, len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if abs(nums[l]) < abs(nums[r]): s, r = nums[r], r - 1
            else: s, l = nums[l], l + 1
            res[i] = s * s
        return res




if __name__ == '__main__':
    """
    One-line quick python tools
    1. For loop - List comprehension
    mylist = [200, 300, 400, 500]
    result = [x for x in mylist if x > 250] 
    print(result) # [300, 400, 500]
    
    Filtering an array using a list
    mylist = [2, 3, 5, 8, 9, 12, 13, 15] 
    result = [x for x in mylist if x % 2 == 0]
    print(result) # [2, 8, 12]
    
    Applying a function to each element in a list using a map function
    print(list(map(lambda a: a + 2, [5, 6, 7, 8, 9, 10])))  
    # print [7, 8, 9, 10, 11, 12]
    
    Applying a function to each element in a list to find the prime number within a range
    print(list(filter(lambda a: all(a % b != 0 for b in range(2, a)), range(2,20))))  
    # print [2, 3, 5, 7, 11, 13, 17, 19]
    
    Removing multiple elements from a list
    mylist = [100, 200, 300, 400, 500]
    del mylist[1::2]  
    print(mylist) # [100, 300, 500]
    
    # Same use of list.index(element) but for all indices in list
    lst = [1, 2, 3, 'Alice', 'Alice']
    indices = [i for i in range(len(lst)) if lst[i]=='Alice']
    print(indices) # [3, 4]
    
    2. While Loop
    x = 0
    while x < 5: print(x); x = x + 1 # 0 1 2 3 4
    
    3. If-elif-else (if else if else)statements
    E = 2   
    print("High") if E == 5 else print("数据STUDIO") if E == 2 else print("Low") # Data STUDIO 
    
    4. Merging dictionaries
    d1, d2 = { 'A': 1, 'B': 2 }, { 'C': 3, 'D': 4 }  
    d1.update(d2) #Method1   
    print(d1) # {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    d3 = {**d1, **d2} #Method2
    print(d3) # {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    Converting a list to a dictionary
    mydict = ["John", "Peter", "Mathew", "Tom"]
    mydict = dict(enumerate(mydict))  
    print(mydict) # {0: 'John', 1: 'Peter', 2: 'Mathew', 3: 'Tom'}
    
    
    5. One-Line Function
    def fun(x): return True if x % 2 == 0 else False #Method1  
    print(fun(2)) # False  
    
    fun = lambda x : x % 2 == 0 #Method2  
    print(fun(2)) # True   
    print(fun(3)) # False
    
    6. Recursion in one-line
    #Fibonaci Single-line recursion example
    def Fib(x): return 1 if x in {0, 1} else Fib(x-1) + Fib(x-2)  
    print(Fib(5)) # 8  
    print(Fib(15)) # 987
    
    7. One line print
    print(*range(1, 5)) # 1 2 3 4
    print(*range(1, 6)) # 1 2 3 4 5
    
    Using asterisk(*)
    print('YES' * 3) # YesYesYes
    
    # Palindrome Python One-Liner
    phrase.find(phrase[::-1])
    
    from itertools import combinations
    # list of all subsets of length r (r = 2 in this example) 
    print(list(combinations([1, 2, 3, 4], 2)))
    # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    

    """
