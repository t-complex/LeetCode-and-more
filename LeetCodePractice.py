import itertools
import collections
import functools
import heapq
import math
from collections import defaultdict, Counter, deque
import sortedcontainers
from UnionFind import UnionFind
from GraphNode import Node
from TreeNode import TreeNode
from ListNode import ListNode
from ListNodeWithRandomPointer import ListNodeWithRandomPointer
from LeetCodeProblems import LeetCodeProblems

class LeetCodePractice():

    """
    External Libraries:
    functools.reduce, ex: reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5).
    count = Counter(nums).most_common(k) #--> this is O(klogn)
    math.comb(i, k), Return the number of ways to choose k items from n items without repetition and without order.
    ljust() method will left align the string, using a specified character (space is default) as the fill character.
    """

    #################### Arrays & Hashing #################### 10 - 59
    """
    - Build a prefix sum
    def fn(arr):
        prefix = [arr[0]]
        for i in range(1, len(arr)):
            prefix.append(prefix[-1] + arr[i])
        return prefix
        
    - Efficient String Building
    def fn(arr): # arr is a list of characters
        ans = []
        for c in arr: ans.append(c)
        return "".join(ans)
    
    - Find num of subarray that fit an exact criteria
    def fn(arr, k):
        counts = defaultdict(int)
        counts[0] = 1
        ans = curr = 0
        for num in arr:
            # do logic to change curr
            ans += counts[curr - k]
            counts[curr] += 1
        return ans
    """

    """
    217. Contains Duplicate
    Given an integer array nums, return true if any value appears at 
    least twice in the array, and return false if every element is distinct.
    Ex1: Input: nums = [1,2,3,1], Output: true
    Ex2: Input: nums = [1,2,3,4], Output: false
    Ex3: Input: nums = [1,1,1,3,3,4,3,2,4,2], Output: true
    DONE
    """
    def contains_duplicate(self, nums: list[int]) -> bool:
        return False

    """
    242. Valid Anagram
    Given two strings s and t, return true if t is an anagram of s, false otherwise.
    Ex1: Input: s = "anagram", t = "nagaram", Output: true
    Ex2: Input: s = "rat", t = "car", Output: false
    """
    def is_anagram(self, s: str, t: str) -> bool:
        return False

    """
    1. Two Sum
    Given an array of integers nums and an integer target,
    return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, 
    and you may not use the same element twice.
    You can return the answer in any order.
    Ex1: Input: nums = [2,7,11,15], target = 9, Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    Ex2: Input: nums = [3,2,4], target = 6, Output: [1,2]
    Ex3: Input: nums = [3,3], target = 6, Output: [0,1]
    DONE
    """
    def two_sum(self, nums: list, target: int) -> list:
        return []

    """
    49. Group Anagrams
    Given an array of strings, group the anagrams together
    You can return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a 
    different word or phrase, typically using all the original letters exactly once.
    Ex1: Input: strs = ["eat","tea","tan","ate","nat","bat"], Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    Ex2: Input: strs = [""], Output: [[""]]
    DONE
    """
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        return [[]]

    """
    347. Top K Frequent Elements
    Given an integer array nums and an integer k,
    return the k most frequent elements. You may return the answer in any order.
    Ex1: Input: nums = [1,1,1,2,2,3], k = 2, Output: [1,2]
    Ex2: Input: nums = [1], k = 1, Output: [1]
    DONE
    """
    """
    Hints: Counter
    """
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        return []

    """
    238. Product of Array Except Self
    Given an integer array nums, return an array answer such that answer[i] is equal to 
    the product of all the elements of nums except nums[i].
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
    You must write an algorithm that runs in O(n) time and without using the division operation.
    Ex1: Input: nums = [1,2,3,4], Output: [24,12,8,6]
    Ex2: Input: nums = [-1,1,0,-3,3], Output: [0,0,9,0,0]
    DONE
    """
    def productExcept(self, nums: list[int]) -> list[int]:
        return []

    """
    36. Valid Sudoku
    Determine if a 9 x 9 Sudoku board is valid. 
    Only the filled cells need to be validated according to the following rules:
    Each row must contain the digits 1-9 without repetition.
    Each column must contain the digits 1-9 without repetition.
    Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
    Note: A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    Only the filled cells need to be validated according to the mentioned rules.
    Ex1: Input: board = 
        [["5","3",".",".","7",".",".",".","."]
        ,["6",".",".","1","9","5",".",".","."]
        ,[".","9","8",".",".",".",".","6","."]
        ,["8",".",".",".","6",".",".",".","3"]
        ,["4",".",".","8",".","3",".",".","1"]
        ,["7",".",".",".","2",".",".",".","6"]
        ,[".","6",".",".",".",".","2","8","."]
        ,[".",".",".","4","1","9",".",".","5"]
        ,[".",".",".",".","8",".",".","7","9"]], Output: true
    Ex2: Input: board = 
        [["8","3",".",".","7",".",".",".","."]
        ,["6",".",".","1","9","5",".",".","."]
        ,[".","9","8",".",".",".",".","6","."]
        ,["8",".",".",".","6",".",".",".","3"]
        ,["4",".",".","8",".","3",".",".","1"]
        ,["7",".",".",".","2",".",".",".","6"]
        ,[".","6",".",".",".",".","2","8","."]
        ,[".",".",".","4","1","9",".",".","5"]
        ,[".",".",".",".","8",".",".","7","9"]], Output: false
    Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. 
    Since there are two 8's in the top left 3x3 sub-box, it is invalid. 
    """
    """
    Hints: 3-dic(set)
    """
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        return True

    """
    659. Encode and Decode Strings
    Design an algorithm to encode a list of strings to a string. 
    The encoded string is then sent over the network and is decoded back to the 
    original list of strings.
    Ex1: Input: strs = ["leet","code","love","you"], Output: ["leet","code","love","you"]
    Ex2: Input: strs = ["we", "say", ":", "yes"], Output: ["we", "say", ":", "yes"]
    """
    def encode(self, strs: list[str]) -> str:
        return ""
    def decode(self, strs: str) -> list[str]:
        return []

    """
    128. Longest Consecutive Sequence
    Given an unsorted array of integers nums, 
    return the length of the longest consecutive elements sequence.
    You must write an algorithm that runs in O(n) time.
    Ex1: Input: nums = [100,4,200,1,3,2], Output: 4
    Ex2: Input: nums = [0,3,7,2,5,8,4,6,0,1], Output: 9
    """
    """
    Hints: set(), check before and after
    """
    def longestConsective(self, nums: list[int]) -> int:
        return 0

    # new picks from 300 list

    """
    1299. Replace Elements with Greatest Element on Right Side
    Given an array arr, replace every element in that array with the greatest element 
    among the elements to its right, and replace the last element with -1.
    After doing so, return the array.
    Ex1: Input: arr = [17,18,5,4,6,1], Output: [18,6,6,6,1,-1]
    Explanation: 
    - index 0 --> the greatest element to the right of index 0 is index 1 (18).
    - index 1 --> the greatest element to the right of index 1 is index 4 (6).
    - index 2 --> the greatest element to the right of index 2 is index 4 (6).
    - index 3 --> the greatest element to the right of index 3 is index 4 (6).
    - index 4 --> the greatest element to the right of index 4 is index 5 (1).
    - index 5 --> there are no elements to the right of index 5, so we put -1.
    Ex2: Input: arr = [400], Output: [-1]
    Explanation: There are no elements to the right of index 0.
    """
    """
    Hints: Backwards
    """
    def replaceElements(self, arr: list[int]) -> list[int]:
        return None

    """
    392. Is Subsequence
    Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
    A subsequence of a string is a new string that is formed from the original string by deleting some 
    (can be none) of the characters without disturbing the relative positions of the remaining characters. 
    (i.e., "ace" is a subsequence of "abcde" while "aec" is not).
    Ex1: Input: s = "abc", t = "ahbgdc", Output: true
    Ex2: Input: s = "axc", t = "ahbgdc", Output: false
    """
    """
    Hints: compare
    """
    def isSubsequence(self, s: str, t: str) -> bool:
        return False

    """
    58. Length of Last Word
    Given a string s consisting of words and spaces, return the length of the last word in the string.
    A word is a maximal substring consisting of non-space characters only.
    Ex1: Input: s = "Hello World", Output: 5
    Explanation: The last word is "World" with length 5.
    Ex2: Input: s = "   fly me   to   the moon  ", Output: 4
    Explanation: The last word is "moon" with length 4.
    Ex3: Input: s = "luffy is still joyboy", Output: 6
    Explanation: The last word is "joyboy" with length 6.
    """
    """
    Hints: split
    """
    def lengthOfLastWord(self, s: str) -> int:
        return 0

    """
    14. Longest Common prefix
    Write a function to find the longest common prefix string amongst an array of strings.
    If there is no common prefix, return an empty string "".
    Ex1: Input: strs = ["flower","flow","flight"], Output: "fl"
    Ex2: Input: strs = ["dog","racecar","car"], Output: ""
    Explanation: There is no common prefix among the input strings.
    """
    """
    Hints: sort check first, last
    """
    def longestCommonPrefix(self, strs: list[str]) -> str:
        return ""

    """
    118. Pascal's Triangle
    Given an integer numRows, return the first numRows of Pascal's triangle.
    In Pascal's triangle, each number is the sum of the two numbers directly above
    Ex1: numRows = 5, Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
    Ex2: numRows = 1, Output: [[1]]
    """
    """
    Hints: 2-fors build row then append
    """
    def generate(self, numRows: int) -> list[list[int]]:
        return None

    """
    68. Text Justification
    Given an array of strings words and a width maxWidth, 
    format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. 
    Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. 
    If the number of spaces on a line does not divide evenly between words, 
    the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left-justified, and no extra space is inserted between words.
    Note: A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array words contains at least one word.
    Ex1: Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
        Output: ["This    is    an",
                 "example  of text",
                 "justification.  " ]
    Ex2: Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
        Output: ["What   must   be",
                 "acknowledgment  ",
                 "shall be        " ]
    Explanation: Note that the last line is "shall be    " instead of "shall     be", 
    because the last line must be left-justified instead of fully-justified.
    Note that the second line is also left-justified because it contains only one word.
    Ex3: Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.",
    "Art","is","everything","else","we","do"], maxWidth = 20
        Output: ["Science  is  what we",
                "understand      well",
                "enough to explain to",
                "a  computer.  Art is",
                "everything  else  we",
                "do                  "]
    """
    """
    Hints: 
    """
    def fullyJustify(self, words: list[int], maxWidth: int) -> list[str]:
        return None

    #################### Two Pointer ######################### 10 - 18
    """
    - Two pointer: one input, opposite ends
    def fn(arr):
        left, right = ans = 0, len(arr) - 1
        while left < right:
            # do some logic here with left and right
            if CONDITION: left += 1
            else:right -= 1
        return ans
        
    - Two pointer: two inputs, exhaust both
    def fn(arr1, arr2):
        i = j = ans = 0
        while i < len(arr1) and j < len(arr2):
            # do some logic here
            if CONDITION: i += 1
            else: j += 1
        while i < len(arr1):
            # do logic
            i += 1
        while j < len(arr2):
            # do logic
            j += 1
        return ans
    """

    """
    125. Valid Palindrome
    A phrase is a palindrome if, after converting all uppercase letters into lowercase letters 
    and removing all non-alphanumeric characters, it reads the same forward and backward. 
    Alphanumeric characters include letters and numbers.
    Given a string s, return true if it is a palindrome, or false otherwise.
    Ex1: Input: s = "A man, a plan, a canal: Panama", Output: true
    Ex2: Input: s = "race a car", Output: false
    """
    """
    Hints: isalnum
    """
    def isPalindrome(self, s: str) -> bool:
        return False

    """
    167. Two Sum II - Input Array Is Sorted
    Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, 
    find two numbers such that they add up to a specific target number. 
    Let these two numbers be numbers[index1] and numbers[index2] 
    where 1 <= index1 < index2 < numbers.length.
    Return the indices of the two numbers, index1 and index2, 
    added by one as an integer array [index1, index2] of length 2.
    The tests are generated such that there is exactly one solution. You may not use the same element twice.
    Your solution must use only constant extra space.
    Ex1: Input: numbers = [2,7,11,15], target = 9, Output: [1,2]
    Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
    Ex2: Input: numbers = [2,3,4], target = 6, Output: [1,3]
    Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
    Ex3: Input: numbers = [-1,0], target = -1, Output: [1,2]
    Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
    """
    """
    Hints: return index+1
    """
    def twoSumInputArray(self, nums: list[int], target: int) -> list[int]:
        return []

    """
    15. 3Sum
    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
    such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    Notice that the solution set must not contain duplicate triplets.
    Ex1: Input: nums = [-1,0,1,2,-1,-4], Output: [[-1,-1,2],[-1,0,1]]
    Explanation: 
        nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
        nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
        nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
        The distinct triplets are [-1,0,1] and [-1,-1,2].
        Notice that the order of the output and the order of the triplets does not matter.
    Ex2: Input: nums = [0,1,1], Output: []
        Explanation: The only possible triplet does not sum up to 0.
    Ex3: Input: nums = [0,0,0], Output: [[0,0,0]]
        Explanation: The only possible triplet sums up to 0.
    """
    """
    Hints: use a for with two pointers, avoid duplicates
    """
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        return None

    """
    11. Container With Most Water
    You are given an integer array height of length n. 
    There are n vertical lines drawn such that the two 
    endpoints of the ith line are (i, 0) and (i, height[i]).
    Find two lines that together with the x-axis form a container, 
    such that the container contains the most water.
    Return the maximum amount of water a container can store.
    Notice that you may not slant the container.
    Ex1: Input: height = [1,8,6,2,5,4,8,3,7], Output: 49
    """
    """
    Hints: max -> min, two-p, res check
    """
    def maxArea(self, height: list[int]) -> int:
        return 0

    """
    42. Trapping Rain water - HARD
    Given n non-negative integers representing an elevation map where the width 
    of each bar is 1, compute how much water it can trap after raining.
    Ex1: Input: height = [0,1,0,2,1,0,1,3,2,1,2,1], Output: 6
    Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
    In this case, 6 units of rain water (blue section) are being trapped.
    Ex2: Input: height = [4,2,0,3,2,5], Output: 9
    """
    """
    Hints: maxl, maxr
    """
    def trap(self, height: list[int]) -> int:
        return 0

    # new Picks from 300 list

    """
    680. Vaild Palindrome II
    Given a string s, 
    return true if the s can be palindrome after deleting at most one character from it.
    Ex1: Input: s = "aba", Output: true
    Ex2: Input: s = "abca", Output: true
    Explanation: You could delete the character 'c'.
    Ex3: Input: s = "abc", Output: false
    """
    """
    Hints: mid, rev, check
    """
    def validPalindrom(self, s: str) -> bool:
        l, r = 0, len(s) -1
        def helper(l, r):
            while l < r:
                if s[l] == s[r]: l, r = l + 1, r - 1
                else: return False
            return True
        while l < r:
            if s[l] == s[r]: l, r = l + 1, r - 1
            else: return helper(l+1, r) or helper(l, r-1)
        return True

    """
    18. 4Sum
    Given an array nums of n integers, return an array of all the unique quadruplets 
    [nums[a], nums[b], nums[c], nums[d]] such that:
        0 <= a, b, c, d < n,  a, b, c, and d are distinct.
    nums[a] + nums[b] + nums[c] + nums[d] == target. You may return the answer in any order.
    Ex1: Input: nums = [1,0,-1,0,-2,2], target = 0, Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
    Ex2: Input: nums = [2,2,2,2,2], target = 8, Output: [[2,2,2,2]]
    """
    """
    Hints: avoid dups, for -3, for -2, while l, r
    """
    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        return None

    """
    88. Merge Sorted Array
    You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, 
    representing the number of elements in nums1 and nums2 respectively.
    Merge nums1 and nums2 into a single array sorted in non-decreasing order.
    The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
    To accommodate this, nums1 has a length of m + n, 
    where the first m elements denote the elements that should be merged, 
    and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
    Ex1: Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3, Output: [1,2,2,3,5,6]
    Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
    The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
    Ex2: Input: nums1 = [1], m = 1, nums2 = [], n = 0, Output: [1]
    Explanation: The arrays we are merging are [1] and []. The result of the merge is [1].
    Ex3: Input: nums1 = [0], m = 0, nums2 = [1], n = 1, Output: [1]
    Explanation: The arrays we are merging are [] and [1]. The result of the merge is [1].
        Note that because m = 0, there are no elements in nums1. 
        The 0 is only there to ensure the merge result can fit in nums1.
    """
    """
    Hints: bisect.in
    """
    def mergeSortedArray(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        return None

    """
    344. Reverse String
    Write a function that reverses a string. The input string is given as an array of characters s.
    You must do this by modifying the input array in-place with O(1) extra memory.
    Ex1: Input: s = ["h","e","l","l","o"], Output: ["o","l","l","e","h"]
    Ex2: Input: s = ["H","a","n","n","a","h"], Output: ["h","a","n","n","a","H"]
    """
    """
    Hints: two pointers
    """
    def reverseString(self, s: list[str]) -> None:
        return None

    """
    198. Rotate Array
    Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.
    Ex1: Input: nums = [1,2,3,4,5,6,7], k = 3, Output: [5,6,7,1,2,3,4]
    Explanation:
        rotate 1 steps to the right: [7,1,2,3,4,5,6]
        rotate 2 steps to the right: [6,7,1,2,3,4,5]
        rotate 3 steps to the right: [5,6,7,1,2,3,4]
    Ex2: Input: nums = [-1,-100,3,99], k = 2, Output: [3,99,-1,-100]
    Explanation: 
        rotate 1 steps to the right: [99,-1,-100,3]
        rotate 2 steps to the right: [3,99,-1,-100]
    """
    def roateArray(self, nums: list[int], k: int) -> None:
        if len(nums) == 1 or k == 0: return
        k = k % len(nums)
        nums[:k], nums[k:] = nums[-k:], nums[:-k]

    """
    1984. Minimum Difference between Highest and Lowest of K scores
    You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student. 
    You are also given an integer k. Pick the scores of any k students from the array so that the 
    difference between the highest and the lowest of the k scores is minimized.
    Return the minimum possible difference.
    Ex1: Input: nums = [90], k = 1, Output: 0
    Explanation: There is one way to pick score(s) of one student:
    - [90]. The difference between the highest and lowest score is 90 - 90 = 0.
    The minimum possible difference is 0.
    Ex2: Input: nums = [9,4,1,7], k = 2, Output: 2
    Explanation: There are six ways to pick score(s) of two students:
    - [9,4,1,7]. The difference between the highest and lowest score is 9 - 4 = 5.
    - [9,4,1,7]. The difference between the highest and lowest score is 9 - 1 = 8.
    - [9,4,1,7]. The difference between the highest and lowest score is 9 - 7 = 2.
    - [9,4,1,7]. The difference between the highest and lowest score is 4 - 1 = 3.
    - [9,4,1,7]. The difference between the highest and lowest score is 7 - 4 = 3.
    - [9,4,1,7]. The difference between the highest and lowest score is 7 - 1 = 6.
    The minimum possible difference is 2.
    """
    """
    Hints: r=k-1
    """
    def minimumDifferences(self, nums: list[int], k:int) -> int:
        nums.sort()
        l, r, res = 0, k - 1, float('inf')
        while r < len(nums):
            res, l, r = min(res, nums[r]-nums[l]), l + 1, r + 1
        return res

    """
    1768. Merge String Alternately
    You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, 
    starting with word1. If a string is longer than the other, append the additional letters 
    onto the end of the merged string. Return the merged string.
    Ex1: Input: word1 = "abc", word2 = "pqr", Output: "apbqcr"
    Explanation: The merged string will be merged as so:
    word1:  a   b   c
    word2:    p   q   r
    merged: a p b q c r
    Ex2: Input: word1 = "ab", word2 = "pqrs", Output: "apbqrs"
    Explanation: Notice that as word2 is longer, "rs" is appended to the end.
    word1:  a   b 
    word2:    p   q   r   s
    merged: a p b q   r   s
    Ex3: Input: word1 = "abcd", word2 = "pq", Output: "apbqcd"
    Explanation: Notice that as word1 is longer, "cd" is appended to the end.
    word1:  a   b   c   d
    word2:    p   q 
    merged: a p b q c   d
    """
    """
    Hints: zip
    """
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return ""

    """
    283. Move Zeroes
    Given an integer array nums, move all 0's to the end of it while maintaining 
    the relative order of the non-zero elements.
    Note that you must do this in-place without making a copy of the array.
    Ex1: Input: nums = [0,1,0,3,12], Output: [1,3,12,0,0]
    Ex2: Input: nums = [0], Output: [0]
    """
    """
    Hints: one-by-one
    """
    def moveZeroes(self, nums: list[int]) -> None:
        return None

    """
    26. Remove Duplicates from Sorted Array
    Given an integer array nums sorted in non-decreasing order, 
    remove the duplicates in-place such that each unique element appears only once. 
    The relative order of the elements should be kept the same. Then return the number of unique elements in nums.
    Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:
    Change the array nums such that the first k elements of nums contain 
    the unique elements in the order they were present in nums initially. 
    The remaining elements of nums are not important as well as the size of nums. Return k.
    Ex1: Input: nums = [1,1,2], Output: 2, nums = [1,2,_]
    Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
        It does not matter what you leave beyond the returned k (hence they are underscores).
    Ex2: Input: nums = [0,0,1,1,1,2,2,3,3,4],  Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
    Explanation: Your function should return k = 5, 
    with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    """
    """
    Hints: 
    """
    def removeDuplicates(self, nums: list[int]) -> int:
        return 0

    """
    80. Remove Duplicates from Sorted Array II
    Given an integer array nums sorted in non-decreasing order, 
    remove some duplicates in-place such that each unique element appears at most twice. 
    The relative order of the elements should be kept the same.
    Since it is impossible to change the length of the array in some languages, 
    you must instead have the result be placed in the first part of the array nums. More formally, 
    if there are k elements after removing the duplicates, 
    then the first k elements of nums should hold the final result. 
    It does not matter what you leave beyond the first k elements.
    Return k after placing the final result in the first k slots of nums.
    Do not allocate extra space for another array. You must do this by modifying the 
    input array in-place with O(1) extra memory.
    Ex1: Input: nums = [1,1,1,2,2,3], Output: 5, nums = [1,1,2,2,3,_]
    Explanation: Your function should return k = 5, 
        with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
        It does not matter what you leave beyond the returned k (hence they are underscores).
    Ex2: Input: nums = [0,0,1,1,1,1,2,3,3], Output: 7, nums = [0,0,1,1,2,3,3,_,_]
    Explanation: Your function should return k = 7, 
        with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    """
    def removeDuplicatesII(self, nums: list[int]) -> int:
        return 0

    """
    1498. Number of Subsequences that satisfy the Given Sum condition
    You are given an array of integers nums and an integer target.
    Return the number of non-empty subsequences of nums such that the sum of the minimum and 
    maximum element on it is less or equal to target. Since the answer may be too large, return it modulo 109 + 7.
    Ex1: Input: nums = [3,5,6,7], target = 9, Output: 4
    Explanation: There are 4 subsequences that satisfy the condition.
        [3] -> Min value + max value <= target (3 + 3 <= 9)
        [3,5] -> (3 + 5 <= 9)
        [3,5,6] -> (3 + 6 <= 9)
        [3,6] -> (3 + 6 <= 9)
    Ex2: Input: nums = [3,3,6,8], target = 10, Output: 6
    Explanation: There are 6 subsequences that satisfy the condition. (nums can have repeated numbers).
        [3] , [3] , [3,3], [3,6] , [3,6] , [3,3,6]
    Ex3: Input: nums = [2,3,3,4,6,7], target = 12, Output: 61
    Explanation: There are 63 non-empty subsequences, two of them do not satisfy the condition ([6,7], [7]).
        Number of valid subsequences (63 - 2 = 61).
    """
    def numSubSeq(self, nums: list[int], target: int) -> int:
        return 0


    #################### Sliding Window ###################### 10 - 14
    """
    def fn(arr):
        left = ans = curr = 0
        for right in range(len(arr)):
            # do logic here to add arr[right] to curr
            while/if WINDOW_CONDITION_BROKEN:
                # remove arr[left] from curr
                left += 1
            # update ans
        return ans
    """

    """
    121. Best Time to Buy and Sell Stock
    You are given an array prices where prices[i] is the price of a given stock on the ith day.
    You want to maximize your profit by choosing a single day to buy one stock 
    and choosing a different day in the future to sell that stock.
    Return the maximum profit you can achieve from this transaction. 
    If you cannot achieve any profit, return 0.
    Ex1: Input: prices = [7,1,5,3,6,4], Output: 5
        Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
        Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
    Ex2: Input: prices = [7,6,4,3,1], Output: 0
        Explanation: In this case, no transactions are done and the max profit = 0.
    """
    """
    Hints: -2, least l, then r-l
    """
    def maxProfile(self, prices: list[int]) -> int:
        return 0

    """
    3. Longest Substring Without Repeating Characters
    Given a string s, find the length of the longest substring without repeating characters.
    Ex1: Input: s = "abcabcbb", Output: 3
    Explanation: The answer is "abc", with the length of 3.
    """
    """
    Hints: -2, set, while
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        return 0

    """
    424. Longest Repeating Character Replacement
    You are given a string s and an integer k. 
    You can choose any character of the string and change it to any other uppercase English character. 
    You can perform this operation at most k times.
    Return the length of the longest substring containing the same letter you can get 
    after performing the above operations.
    Ex1: Input: s = "ABAB", k = 2, Output: 4
        Explanation: Replace the two 'A's with two 'B's or vice versa.
    Ex2: Input: s = "AABABBA", k = 1, Output: 4
        Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
        The substring "BBBB" has the longest repeating letters, which is 4.
    """
    """
    Hints: -2, map
    """
    def characterReplacement(self, s: str, k: int) -> int:
        return 0

    """
    567. Permutation in String
    Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.
    In other words, return true if one of s1's permutations is the substring of s2.
    Ex1: Input: s1 = "ab", s2 = "eidbaooo", Output: true
        Explanation: s2 contains one permutation of s1 ("ba").
    Ex2: Input: s1 = "ab", s2 = "eidboaoo", Output: false
    """
    """
    Hints: -1, built counter for s1 and an empty counter, add char, dec s2[i-n] in the sec counter
    """
    def checkInclusion(self, s1: str, s2: str) -> bool:
        return False

    """
    76. Minimum Window Substring - HARD
    Given two strings s and t of lengths m and n respectively, 
    return the minimum window substring of s such that every character in t 
    (including duplicates) is included in the window. 
    If there is no such substring, return the empty string "".
    The testcases will be generated such that the answer is unique.
    A substring is a contiguous sequence of characters within the string.
    Ex1: Input: s = "ADOBECODEBANC", t = "ABC", Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    """
    """
    Hints: Need to memorize 
    """
    def minWindowSubstring(self, s: str, t: str) -> str:
        return None

    """
    239. Sliding Window Maximum - HARD
    You are given an array of integers nums, there is a sliding window of size k 
    which is moving from the very left of the array to the very right. 
    You can only see the k numbers in the window. 
    Each time the sliding window moves right by one position.
    Return the max sliding window.
    Ex1: Input: nums = [1,3,-1,-3,5,3,6,7], k = 3, Output: [3,3,5,5,6,7]
    """
    """
    Hints:
    """
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        return None

    # new Picks from 300 list

    """
    219. Contains Duplicate II
    Given an integer array nums and an integer k, return true if there are 
    two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
    Ex1: Input: nums = [1,2,3,1], k = 3, Output: true
    Ex2: Input: nums = [1,0,1,1], k = 1, Output: true
    Ex3: Input: nums = [1,2,3,1,2,3], k = 2, Output: false
    """
    """
    Hints: -1, set doesnt exceed k+1, otherwise remove
    """
    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        return False

    """
    1343. Number of Sub-arrays of size K and average Greater than or Equal to Threshold
    Given an array of integers arr and two integers k and threshold, 
    return the number of sub-arrays of size k and average greater than or equal to threshold.
    Ex1: Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4, Output: 3
    Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. 
    All other sub-arrays of size 3 have averages less than 4 (the threshold).
    Ex2: Input: arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5, Output: 6
    Explanation: The first 6 sub-arrays of size 3 have averages greater than 5. 
    Note that averages are not integers.
    """
    """
    Hints: -1, curSum, add, check, subtract
    """
    def numOfSubarrays(self, arr: list[int], k: int, threshold: int) -> int:
        return 0

    """
    1838. Frequency of the Most Frequent Element
    The frequency of an element is the number of times it occurs in an array.
    You are given an integer array nums and an integer k. In one operation, 
    you can choose an index of nums and increment the element at that index by 1.
    Return the maximum possible frequency of an element after performing at most k operations.
    Ex1: Input: nums = [1,2,4], k = 5, Output: 3
    Explanation: Increment the first element three times and the second element 
    two times to make nums = [4,4,4]. 4 has a frequency of 3.
    Ex2: Input: nums = [1,4,8,13], k = 5, Output: 2
    Explanation: There are multiple optimal solutions:
    - Increment the first element three times to make nums = [4,4,8,13]. 4 has a frequency of 2.
    - Increment the second element four times to make nums = [1,8,8,13]. 8 has a frequency of 2.
    - Increment the third element five times to make nums = [1,4,13,13]. 13 has a frequency of 2.
    Ex3: Input: nums = [3,9,6], k = 2, Output: 1
    """
    """
    Hints: -2, sort, curSum
    """
    def maxFrequency(self, nums: list[int], k: int) -> int:
        return 0

    """
    209. Minimum Size Subarray Sum
    Given an array of positive integers nums and a positive integer target, return the minimal length of a 
    subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.
    Ex1: Input: target = 7, nums = [2,3,1,2,4,3], Output: 2
    Explanation: The subarray [4,3] has the minimal length under the problem constraint.
    Ex2: Input: target = 4, nums = [1,4,4], Output: 1
    Ex3: Input: target = 11, nums = [1,1,1,1,1,1,1,1], Output: 0
    """
    """
    Hints: -2, cursum, minlen=inf, -sum >= target
    """
    def minSubArrayLen(self, nums: list[int], target: int) -> int:
        return 0

    """
    658. Find K closest Elements
    Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. 
    The result should also be sorted in ascending order.
    An integer a is closer to x than an integer b if:
        |a - x| < |b - x|, or
        |a - x| == |b - x| and a < b
    Ex1: Input: arr = [1,2,3,4,5], k = 4, x = 3, Output: [1,2,3,4]
    Ex2: Input: arr = [1,2,3,4,5], k = 4, x = -1, Output: [1,2,3,4]
    """
    """
    Hints: -2, mid, check
    """
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        return None

    """
    1456. Maximum Number of Vowels in a Substring of Given Length
    Given a string s and an integer k, return the maximum number of vowel letters in any 
    substring of s with length k. Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.
    Ex1: Input: s = "abciiidef", k = 3, Output: 3, Explanation: The substring "iii" contains 3 vowel letters.
    Ex2: Input: s = "aeiou", k = 2, Output: 2, Explanation: Any substring of length 2 contains 2 vowels.
    Ex3: Input: s = "leetcode", k = 3, Output: 2, Explanation: "lee", "eet" and "ode" contain 2 vowels.
    """
    """
    Hints: -2, cur
    """
    def maxVowels(self, s: str, k: int) -> int:
        return 0

    """
    904. Fruit Into Baskets
    You are visiting a farm that has a single row of fruit trees arranged from left to right. 
    The trees are represented by an integer array fruits where fruits[i] is the type of 
    fruit the ith tree produces. You want to collect as much fruit as possible. 
    However, the owner has some strict rules that you must follow:
    You only have two baskets, and each basket can only hold a single type of fruit. 
    There is no limit on the amount of fruit each basket can hold.
    Starting from any tree of your choice, you must pick exactly one fruit from every tree 
    (including the start tree) while moving to the right. 
    The picked fruits must fit in one of your baskets.
    Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
    Given the integer array fruits, return the maximum number of fruits you can pick.
    Ex1: Input: fruits = [1,2,1], Output: 3, Explanation: We can pick from all 3 trees.
    Ex2: Input: fruits = [0,1,2,2], Output: 3, Explanation: We can pick from trees [1,2,2].
        If we had started at the first tree, we would only pick from trees [0,1].
    Ex3: Input: fruits = [1,2,3,2,2], Output: 4, Explanation: We can pick from trees [2,3,2,2].
        If we had started at the first tree, we would only pick from trees [1,2].
    """
    """
    Hints:
    """
    def totalFruit(self, fruits: list[int]) -> int:
        res, l, mapp = 0, 0, collections.defaultdict(int)
        for r in range(1, len(fruits)):
            if len(mapp) <= 2: mapp[fruits[r]] += 1

        return res
    """
    1888. Minimum Number of Flips to make the Binary String Alternating
    You are given a binary string s. You are allowed to perform two types of operations on the string in any sequence:
    Type-1: Remove the character at the start of the string s and append it to the end of the string.
    Type-2: Pick any character in s and flip its value, i.e., if its value is '0' it becomes '1' and vice-versa.
    Return the minimum number of type-2 operations you need to perform such that s becomes alternating.
    The string is called alternating if no two adjacent characters are equal.
    For example, the strings "010" and "1010" are alternating, while the string "0100" is not.
    Ex1: Input: s = "111000", Output: 2, Explanation: Use the first operation two times to make s = "100011".
        Then, use the second operation on the third and sixth elements to make s = "101010".
    Ex2: Input: s = "010", Output: 0, Explanation: The string is already alternating.
    Ex3: Input: s = "1110", Output: 1
        Explanation: Use the second operation on the second element to make s = "1010".
    """
    """
    Hints:
    """
    def minFlips(self, s: str) -> int: return 0

    #################### Stacks ############################## 10 - 19
    """
    - Monotonic Increasing Stack
    def fn(arr):
        stack, ans = [], 0
        for num in arr:
            # for monotonic decreasing, just flip the > to <
            while stack and stack[-1] > num:
                # do logic
                stack.pop()
            stack.append(num)
        
        return ans
    """

    """
    20. Valid Parenthese
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
    determine if the input string is valid. An input string is valid if:
        Open brackets must be closed by the same type of brackets.
        Open brackets must be closed in the correct order.
        Every close bracket has a corresponding open bracket of the same type.
    Ex1: Input: s = "()[]{}", Output: true
    Ex2: Input: s = "(]", Output: false
    """
    """
    Hints: map, check stack
    """
    def isValid(self, s: str) -> bool:
        return False

    """
    155. Min Stack
    CHECK MinStack CLASS
    """

    """
    150. Evaluate Reverse Polish Notation
    You are given an array of strings tokens that 
    represents an arithmetic expression in a Reverse Polish Notation.
    Evaluate the expression. Return an integer that represents the value of the expression.
    Note that: The valid operators are '+', '-', '*', and '/'.
        Each operand may be an integer or another expression.
        The division between two integers always truncates toward zero.
        There will not be any division by zero.
        The input represents a valid arithmetic expression in a reverse polish notation.
        The answer and all the intermediate calculations can be represented in a 32-bit integer.
    Ex1: Input: tokens = ["2","1","+","3","*"], Output: 9 - Explanation: ((2 + 1) * 3) = 9
    Ex2: Input: tokens = ["4","13","5","/","+"], Output: 6 - Explanation: (4 + (13 / 5)) = 6
    Ex3: Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"], Output: 22
    Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
        = ((10 * (6 / (12 * -11))) + 17) + 5
        = ((10 * (6 / -132)) + 17) + 5
        = ((10 * 0) + 17) + 5
        = (0 + 17) + 5
        = 17 + 5
        = 22
    """
    """
    Hints: add #s to stack, check the oper, then append (b-a)
    """
    def evalRPN(self, tokens: list[str]) -> int:
        return 0

    """
    22. Generate Parentheses
    Given n pairs of parentheses, 
    write a function to generate all combinations of well-formed parentheses.
    Ex1: Input: n = 3, Output: ["((()))","(()())","(())()","()(())","()()()"]
    """
    """
    Hints: COMBINATIONS -> Backtrack(recursion) on open & close < n, open < n, close < open
    """
    def generateParenthesis(self, n: int) -> list[str]:
        return None

    """
    739. Daily Temperatures
    Given an array of integers temperatures represents the daily temperatures, 
    return an array answer such that answer[i] is the number of days 
    you have to wait after the ith day to get a warmer temperature. 
    If there is no future day for which this is possible, keep answer[i] == 0 instead.
    Ex1: Input: temperatures = [73,74,75,71,69,72,76,73], Output: [1,1,4,2,1,1,0,0]
    Ex2: Input: temperatures = [30,40,50,60], Output: [1,1,1,0]
    Ex3: Input: temperatures = [30,60,90], Output: [1,1,0] 
    """
    """
    Hints: res, stack, res=[temp, ind]
    """
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        return None

    """
    853. Car Fleet
    There are n cars going to the same destination along a one-lane road. 
    The destination is target miles away. You are given two integer array position and speed, 
    both of length n, where position[i] is the position of the ith car and speed[i] is 
    the speed of the ith car (in miles per hour). A car can never pass another car ahead of it, 
    but it can catch up to it and drive bumper to bumper at the same speed. The faster car will slow down to match the slower car's speed. 
    The distance between these two cars is ignored  (i.e., they are assumed to have the same position).
    A car fleet is some non-empty set of cars driving at the same position and same speed. 
    Note that a single car is also a car fleet.
    If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.
    Return the number of car fleets that will arrive at the destination.
    Ex1: Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3], Output: 3
    Explanation:
    The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12.
    The car starting at 0 does not catch up to any other car, so it is a fleet by itself.
    The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.
    Note that no other cars meet these fleets before the destination, so the answer is 3.
    """
    """
    Hints: zip, append()
    """
    def carFleet(self, target: int, position: list[int], speed: list[int]) -> int:
        return 0

    """
    84. Largest Rectangle in Histogram  - HARD
    Given an array of integers heights representing the histogram's bar height 
    where the width of each bar is 1, 
    return the area of the largest rectangle in the histogram.
    Ex1: Input: heights = [2,1,5,6,2,3], Output: 10
    Explanation: The above is a histogram where width of each bar is 1.
    The largest rectangle is shown in the red area, which has an area = 10 units.
    """
    """
    Hints: stack[i, h]
    """
    def largestRectangleArea(self, heights: list[int]) -> int:
        return 0

    # new Picks from 300 list

    """
    225. MyStack
    CHECK MyStack CLASS
    """

    """
    682. Baseball Game
    You are keeping the scores for a baseball game with strange rules. 
    At the beginning of the game, you start with an empty record.
    You are given a list of strings operations, where operations[i] is the ith operation 
    you must apply to the record and is one of the following:
    An integer x.
        Record a new score of x.
        '+'.
        Record a new score that is the sum of the previous two scores.
        'D'.
        Record a new score that is the double of the previous score.
        'C'.
        Invalidate the previous score, removing it from the record.
    Return the sum of all the scores on the record after applying all the operations.
    The test cases are generated such that the answer and all intermediate calculations 
    fit in a 32-bit integer and that all operations are valid.
    Ex1: Input: ops = ["5","2","C","D","+"], Output: 30
    Explanation:
        "5" - Add 5 to the record, record is now [5].
        "2" - Add 2 to the record, record is now [5, 2].
        "C" - Invalidate and remove the previous score, record is now [5].
        "D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
        "+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
        The total sum is 5 + 10 + 15 = 30.
    Ex2: Input: ops = ["5","-2","4","C","D","9","+","+"], Output: 27
    Explanation:
        "5" - Add 5 to the record, record is now [5].
        "-2" - Add -2 to the record, record is now [5, -2].
        "4" - Add 4 to the record, record is now [5, -2, 4].
        "C" - Invalidate and remove the previous score, record is now [5, -2].
        "D" - Add 2 * -2 = -4 to the record, record is now [5, -2, -4].
        "9" - Add 9 to the record, record is now [5, -2, -4, 9].
        "+" - Add -4 + 9 = 5 to the record, record is now [5, -2, -4, 9, 5].
        "+" - Add 9 + 5 = 14 to the record, record is now [5, -2, -4, 9, 5, 14].
        The total sum is 5 + -2 + -4 + 9 + 5 + 14 = 27.
    Ex3: Input: ops = ["1","C"], Output: 0
    Explanation:
        "1" - Add 1 to the record, record is now [1].
        "C" - Invalidate and remove the previous score, record is now [].
        Since the record is empty, the total sum is 0.
    """
    """
    Hints: make sure to append ints 
    """
    def calPoints(self, operations: list[str]) -> int:
        return 0

    """
    2390. Removing Stars from a String
    You are given a string s, which contains stars *. In one operation, you can:
        Choose a star in s.
        Remove the closest non-star character to its left, as well as remove the star itself.
    Return the string after all stars have been removed.
    Note: The input will be generated such that the operation is always possible.
    It can be shown that the resulting string will always be unique.
    Ex1: Input: s = "leet**cod*e", Output: "lecoe"
    Explanation: Performing the removals from left to right:
    - The closest character to the 1st star is 't' in "leet**cod*e". s becomes "lee*cod*e".
    - The closest character to the 2nd star is 'e' in "lee*cod*e". s becomes "lecod*e".
    - The closest character to the 3rd star is 'd' in "lecod*e". s becomes "lecoe".
    There are no more stars, so we return "lecoe".
    Ex2: Input: s = "erase*****", Output: ""
    Explanation: The entire string is removed, so we return an empty string.
    """
    def removeStars(self, s: str) -> str:
        return ""

    """
    901. Design an algorithm that collects daily price quotes for some stock and 
    returns the span of that stock's price for the current day.
    The span of the stock's price in one day is the maximum number of consecutive days 
    (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.
    For example, if the prices of the stock in the last four days is [7,2,1,2] 
    and the price of the stock today is 2, then the span of today is 4 because starting from today, 
    the price of the stock was less than or equal 2 for 4 consecutive days.
    Also, if the prices of the stock in the last four days is [7,34,1,2] 
    and the price of the stock today is 8, then the span of today is 3 because starting from today, 
    the price of the stock was less than or equal 8 for 3 consecutive days.
    Implement the StockSpanner class:
    StockSpanner() Initializes the object of the class.
    int next(int price) Returns the span of the stock's price given that today's price is price.
    Ex1: Input ["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
    [[], [100], [80], [60], [70], [60], [75], [85]], Output [null, 1, 1, 1, 2, 1, 4, 6]
    Explanation
    StockSpanner stockSpanner = new StockSpanner();
    stockSpanner.next(100); // return 1
    stockSpanner.next(80);  // return 1
    stockSpanner.next(60);  // return 1
    stockSpanner.next(70);  // return 2
    stockSpanner.next(60);  // return 1
    stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
    stockSpanner.next(85);  // return 6
    """
    """
    Hints: def __init__(self): self.stack = [] # (price, span)
    """
    def nextStockSpanner(self, stack, price: int) -> int:
        return 0

    """
    946. Validate Stack Sequences
    Given two integer arrays pushed and popped each with distinct values, 
    return true if this could have been the result of a sequence of push and pop 
    operations on an initially empty stack, or false otherwise.
    Ex1: Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1], Output: true
    Explanation: We might do the following sequence:
        push(1), push(2), push(3), push(4),
        pop() -> 4,
        push(5),
        pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
    Ex2: Input: pushed = [1,2,3,4,5], popped = [4,3,5,1,2], Output: false
    Explanation: 1 cannot be popped before 2.
    """
    """
    Hints: use stack for indices in push, stack[-1] == pop[j]
    """
    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        return False

    #################### Binary Search ####################### 10 - 24
    """
    - Binary Search
    def fn(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                # do something
                return
            if arr[mid] > target: right = mid - 1
            else: left = mid + 1
        # left is the insertion point
        return left
    
    - Binary Search: duplicate elements, left-most insertion point
    def fn(arr, target):
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] >= target: right = mid
            else: left = mid + 1
        return left
        
    - Binary Search: duplicate elemets, right-most insertion point
    def fn(arr, target):
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > target: right = mid
            else: left = mid + 1
        return left
    
    - Binary Search: For greedy problems
    def fn(arr):
        def check(x):
            # this function is implemented depending on the problem
            return BOOLEAN
        left, right = MINIMUM_POSSIBLE_ANSWER, MAXIMUM_POSSIBLE_ANSWER
        while left <= right:
            mid = (left + right) // 2                     // if you are looking for a max
            if check(mid): right = mid - 1               if check(mid): left = mid + 1
            else: left = mid + 1                         else: right = mid - 1
    return left                                          return right
    """

    """
    704. Binary Search
    Given an array of integers nums which is sorted in ascending order, 
    and an integer target, write a function to search target in nums. 
    If target exists, then return its index. Otherwise, return -1.
    You must write an algorithm with O (log n) runtime complexity.
    Ex1: Input: nums = [-1,0,3,5,9,12], target = 9, Output: 4
    Explanation: 9 exists in nums and its index is 4
    Ex2: Input: nums = [-1,0,3,5,9,12], target = 2, Output: -1
    Explanation: 2 does not exist in nums so return -1
    """
    """
    Hints: l <= r, 
    """
    def binarySearch(self, nums: list[int], target: int) -> int:
        return -1

    """
    74. Search a 2D Matrix
    You are given an m x n integer matrix with the following two properties:
    Each row is sorted in non-decreasing order.
    The first integer of each row is greater than the last integer of the previous row.
    Given an integer target, return true if target is in matrix or false otherwise.
    You must write a solution in O(log(m * n)) time complexity.
    Ex1: Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3, Output: true
    Ex2: Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13, Output: false
    """
    """
    Hints: l=0,r=n*m-1, binary search while using divmod for row, col
    """
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        return False

    """
    875. Koko Eating Bananas
    Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. 
    The guards have gone and will come back in h hours.
    Koko can decide her bananas-per-hour eating speed of k. 
    Each hour, she chooses some pile of bananas and eats k bananas from that pile. 
    If the pile has less than k bananas, she eats all of them instead and 
    will not eat any more bananas during this hour.
    Koko likes to eat slowly but still wants to finish eating all 
    the bananas before the guards return.
    Return the minimum integer k such that she can eat all the bananas within h hours.
    Ex1: Input: piles = [3,6,7,11], h=8, output = 4
    Ex2: Input: piles = [30,11,23,4,20], h = 5, Output: 30
    Ex3: Input: piles = [30,11,23,4,20], h = 6, Output: 23
    """
    """
    Hints: l=1, r=max(piles), binary search on the nums not index, 
    (to get the time, for p in piles, +=ceil p/m)
    """
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        return 0

    """
    153. Find Minimum in Rotated Sorted Array
    Suppose an array of length n sorted in ascending order is rotated between 1 and n times. 
    For example, the array nums = [0,1,2,4,5,6,7] might become:
        [4,5,6,7,0,1,2] if it was rotated 4 times.
        [0,1,2,4,5,6,7] if it was rotated 7 times.
    Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array
    [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
    Given the sorted rotated array nums of unique elements, return the minimum element of this array.
    You must write an algorithm that runs in O(log n) time.
    Ex1: Input: nums = [3,4,5,1,2], Output: 1
    Ex2: Input: nums = [4,5,6,7,0,1,2], Output: 0
    """
    """
    Hints: set curmin to inf, ret min()
    """
    def findMin(self, nums: list[int]) -> int:
        return 0

    """
    33. Search in Rotated Sorted Array
    There is an integer array nums sorted in ascending order (with distinct values).
    Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) 
    such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    Given the array nums after the possible rotation and an integer target, 
    return the index of target if it is in nums, or -1 if it is not in nums.
    You must write an algorithm with O(log n) runtime complexity.
    Ex1: Input: nums = [4,5,6,7,0,1,2], target = 0, Output: 4
    Ex2: Input: nums = [4,5,6,7,0,1,2], target = 3, Output: -1
    """
    """
    Hints: Binary Search and nested ifs m-l, m-tar | l-tar, nested ifs for the opposite
    """
    def searchInRotatedSortedArray(self, nums: list[int], target: int) -> int:
        return -1

    """
    981. Time Based Key-Value Store - TimeMap CLASS
    """

    """
    4. Median of Two Sorted Arrays - HARD
    Given two sorted arrays nums1 and nums2 of size m and n respectively, 
    return the median of the two sorted arrays.
    The overall run time complexity should be O(log (m+n)).
    Ex1: Input: nums1 = [1,3], nums2 = [2], Output: 2.00000
    Explanation: merged array = [1,2,3] and median is 2.
    Ex2: Input: nums1 = [1,2], nums2 = [3,4], Output: 2.50000
    Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
    """
    """
    Hints: You got this
    """
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        return 0.0

    # new Picks from 300 list

    """
    35. Search Insert Position
    Given a sorted array of distinct integers and a target value, return the index if the target is found. 
    If not, return the index where it would be if it were inserted in order.
    You must write an algorithm with O(log n) runtime complexity.
    Ex1: Input: nums = [1,3,5,6], target = 5, Output: 2
    Ex2: Input: nums = [1,3,5,6], target = 2, Output: 1
    Ex3: Input: nums = [1,3,5,6], target = 7, Output: 4
    """
    """
    Hints: binary search, then check mid
    """
    def searchInsert(self, nums: list[int], target: int) -> int:
        return 0

    """
    374. Guess Number Higher or Lower
    We are playing the Guess Game. The game is as follows:
    I pick a number from 1 to n. You have to guess which number I picked.
    Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.
    You call a pre-defined API int guess(int num), which returns three possible results:
        -1: Your guess is higher than the number I picked (i.e. num > pick).
        1: Your guess is lower than the number I picked (i.e. num < pick).
        0: your guess is equal to the number I picked (i.e. num == pick).
    Return the number that I picked.
    Ex1: Input: n = 10, pick = 6, Output: 6
    Ex2: Input: n = 1, pick = 1, Output: 1
    Ex3: Input: n = 2, pick = 1, Output: 1
    """
    """
    Hints: binary search, call api
    """
    def guessNumber(self, n: int) -> int:
        return 0

    """
    540. Single Element in a Sorted Array
    You are given a sorted array consisting of only integers where every element appears exactly twice, 
    except for one element which appears exactly once. Return the single element that appears only once.
    Your solution must run in O(log n) time and O(1) space.
    Ex1: Input: nums = [1,1,2,3,3,4,4,8,8], Output: 2, l = 0, mid=4 , r=8
    Ex2: Input: nums = [3,3,7,7,10,11,11], Output: 10
    """
    """
    Hints: 
    """
    def singleNonDuplicate(self, nums: list[int]) -> int:
        return 0

    """
    81. Search in Rotated Sorted Array II
    There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).
    Before being passed to your function, nums is rotated at an unknown pivot 
    index k (0 <= k < nums.length) such that the resulting 
    array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].
    Given the array nums after the rotation and an integer target, 
    return true if target is in nums, or false if it is not in nums.
    You must decrease the overall operation steps as much as possible.
    Ex1: Input: nums = [2,5,6,0,0,1,2], target = 0, Output: true
    Ex2: Input: nums = [2,5,6,0,0,1,2], target = 3, Output: false
    """
    """
    Hints: 
    """
    def searchInRotatedSortedArrayII(self, nums: list[int], target: int) -> bool:
        return False

    """
    2616. Minimize the Max difference of Pairs
    You are given a 0-indexed integer array nums and an integer p. 
    Find p pairs of indices of nums such that the maximum difference amongst all the pairs is minimized. 
    Also, ensure no index appears more than once amongst the p pairs.
    Note that for a pair of elements at the index i and j, the difference of 
    this pair is |nums[i] - nums[j]|, where |x| represents the absolute value of x.
    Return the minimum maximum difference among all p pairs. We define the maximum of an empty set to be zero.
    Ex1: Input: nums = [10,1,2,7,1,3], p = 2, Output: 1
    Explanation: The first pair is formed from the indices 1 and 4, 
    and the second pair is formed from the indices 2 and 5. 
    The maximum difference is max(|nums[1] - nums[4]|, |nums[2] - nums[5]|) = max(0, 1) = 1. Therefore, we return 1.
    Ex2: Input: nums = [4,2,1,2], p = 1, Output: 0
    Explanation: Let the indices 1 and 3 form a pair. 
    The difference of that pair is |2 - 2| = 0, which is the minimum we can attain.
    """
    def minimizeMax(self, nums: list[int], p: int) -> int:
        return 0

    #################### LinkedList ########################## 10 - 28
    """
    - Fast and Slow Pointer
    def fn(head):
        slow = head
        fast = head
        ans = 0
        while fast and fast.next:
            # do logic
            slow = slow.next
            fast = fast.next.next
        return ans
        
    - Reversing a linkedlist
    def fn(head):
        curr = head
        prev = None
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node 
        return prev
    """

    """
    206. Reverse Linked List
    Given the head of a singly linked list, reverse the list, and return the reversed list.
    Ex1: Input: head = [1,2,3,4,5], Output: [5,4,3,2,1]
    Ex2: Input: head = [1,2], Output: [2,1]
    """
    def reverseLList(self, head: ListNode) -> ListNode:
        return None

    """
    21. Merge Two Sorted Lists
    You are given the heads of two sorted linked lists list1 and list2.
    Merge the two lists into one sorted list. 
    The list should be made by splicing together the nodes of the first two lists.
    Return the head of the merged linked list. 
    Ex1: Input: list1 = [1,2,4], list2 = [1,3,4], Output: [1,1,2,3,4,4]
    """
    def mergeTwoLinkedList(self, list1: ListNode, list2: ListNode) -> ListNode:
        return None

    """
    141. Linked List Cycle
    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    There is a cycle in a linked list if there is some node in the list that can be reached again 
    by continuously following the next pointer. 
    Internally, pos is used to denote the index of the node that tail's next pointer is connected to. 
    Note that pos is not passed as a parameter.
    Return true if there is a cycle in the linked list. Otherwise, return false.
    Ex1: Input: head = [3,2,0,-4], pos = 1, Output: true
    Explanation: There is a cycle in the linked list, 
    where the tail connects to the 1st node (0-indexed).
    Ex2: Input: head = [1,2], pos = 0 - Output: true
    Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
    Ex3: Input: head = [1], pos = -1 - Output: false
    Explanation: There is no cycle in the linked list.
    """
    def hasCycle(self, head: ListNode) -> bool:
        return None

    """
    143. Reorder List
    You are given the head of a singly linked-list. The list can be represented as:
    L0 → L1 → … → Ln - 1 → Ln. Reorder the list to be on the following form:
    L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → … You may not modify the values in the list's nodes. 
    Only nodes themselves may be changed.
    Ex1: Input: head = [1,2,3,4], Output: [1,4,2,3]
    Ex2: Input: head = [1,2,3,4,5], Output: [1,5,2,4,3]
    """
    def reorderLinkedList(self, head: ListNode) -> None:
        return None

    """
    19. Remove Nth Node from End of the list
    Given the head of a linked list, 
    remove the nth node from the end of the list and return its head.
    Ex1: Input: head = [1,2,3,4,5], n = 2 - Output: [1,2,3,5]
    Ex2: Input: head = [1], n = 1, Output: []
    Ex3: Input: head = [1,2], n = 1, Output: [1]
    [1,2,3,4,5,6,7,8,9,10]
    """
    """
    Hints: move pointer by n, then use another pointer and move them together then remove
    """
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        return None

    """
    138. Copy List with Random Pointer
    A linked list of length n is given such that each node contains an additional random pointer, 
    which could point to any node in the list, or null.
    Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, 
    where each new node has its value set to the value of its corresponding original node. 
    Both the next and random pointer of the new nodes should point to new nodes in the copied list 
    such that the pointers in the original list and copied list represent the same list state. 
    None of the pointers in the new list should point to nodes in the original list.
    For example, if there are two nodes X and Y in the original list, where X.random --> Y, 
    then for the corresponding two nodes x and y in the copied list, x.random --> y.
    Return the head of the copied linked list.
    The linked list is represented in the input/output as a list of n nodes. 
    Each node is represented as a pair of [val, random_index] where:
    val: an integer representing Node.val
    random_index: the index of the node (range from 0 to n-1) 
    that the random pointer points to, or null if it does not point to any node.
    Your code will only be given the head of the original linked list.
    Ex1: Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]], Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    Ex2: Input: head = [[1,1],[2,1]], Output: [[1,1],[2,1]]
    Ex3: Input: head = [[3,null],[3,0],[3,null]], Output: [[3,null],[3,0],[3,null]]
    """
    """
    Hints: need a hashmap, build the hashmap, then build the deep copy
    """
    def copyRandomList(self, head: ListNodeWithRandomPointer) -> ListNodeWithRandomPointer:
        return None

    """
    2. Add Two Numbers
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in reverse order, and each of their nodes contains a single digit. 
    Add the two numbers and return the sum as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Ex1: Input: l1 = [2,4,3], l2 = [5,6,4], Output: [7,0,8] - Explanation: 342 + 465 = 807.
    Ex2: Input: l1 = [0], l2 = [0], Output: [0]
    Ex3: Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9], Output: [8,9,9,9,0,0,0,1]
    """
    """
    Hints: or, divmod, create, next
    """
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        return None

    """
    287. Find the Duplicate Number
    Given an array of integers nums containing n + 1 integers 
    where each integer is in the range [1, n] inclusive.
    There is only one repeated number in nums, return this repeated number.
    You must solve the problem without modifying the array nums and uses 
    only constant extra space.
    Ex1: Input: nums = [1,3,4,2,2], Output: 2
    Ex2: Input: nums = [3,1,3,4,2], Output: 3
    """
    """
    Hints:
    """
    def findDuplicate(self, nums: list[int]) -> int:
        res = sum(nums)
        return 0

    """
    23. Merge k sorted Lists - HARD
    You are given an array of k linked-lists lists, 
    each linked-list is sorted in ascending order.
    Merge all the linked-lists into one sorted linked-list and return it.
    Ex1: Input: lists = [[1,4,5],[1,3,4],[2,6]], Output: [1,1,2,3,4,4,5,6]
    Explanation: The linked-lists are:
    [
      1->4->5,
      1->3->4,
      2->6
    ]
    merging them into one sorted list:
    1->1->2->3->4->4->5->6
    """
    """
    Hints:
    """
    def mergeKList(self, lists: list[ListNode]) -> ListNode:
        return None

    """
    25. Reverse Nodes in k-Group - HARD
    Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
    k is a positive integer and is less than or equal to the length of the linked list. 
    If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
    You may not alter the values in the list's nodes, only nodes themselves may be changed.
    Ex1: Input: head = [1,2,3,4,5], k = 2, Output: [2,1,4,3,5]
    Ex2: Input: head = [1,2,3,4,5], k = 3, Output: [3,2,1,4,5]
    """
    """
    Hints:
    """
    def reverseKGroup(self, lists: list[ListNode], k: int) -> ListNode:
        return None

    # new Picks from 300 list

    """
    234. Palindrome Linked List
    Given the head of a singly linked list, return true if it is a palindrome or false otherwise.
    Ex1: Input: head = [1,2,2,1], Output: true
    Ex2: Input: head = [1,2], Output: false
    [1,2,3,4,   4,3,2,1] slow=4, fast=1
    """
    """
    Hints: slow, fast, reverse second half then compare
    """
    def isPalindromeLL(self, head: ListNode) -> bool:
        return False

    """
    2130. Maximum Twin Sum of a Linked List
    In a linked list of size n, where n is even, the ith node (0-indexed) of the linked list is known as 
    the twin of the (n-1-i)th node, if 0 <= i <= (n / 2) - 1.
    For example, if n = 4, then node 0 is the twin of node 3, and node 1 is the twin of node 2. 
    These are the only nodes with twins for n = 4.
    The twin sum is defined as the sum of a node and its twin.
    Given the head of a linked list with even length, return the maximum twin sum of the linked list.
    Ex1: Input: head = [5,4,2,1], Output: 6
    Explanation:
        Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
        There are no other nodes with twins in the linked list.
        Thus, the maximum twin sum of the linked list is 6. 
    Ex2: Input: head = [4,2,2,3], Output: 7
    Explanation:
        The nodes with twins present in this linked list are:
        - Node 0 is the twin of node 3 having a twin sum of 4 + 3 = 7.
        - Node 1 is the twin of node 2 having a twin sum of 2 + 2 = 4.
        Thus, the maximum twin sum of the linked list is max(7, 4) = 7.
    Ex3: Input: head = [1,100000], Output: 100001
    Explanation:
        There is only one node with a twin in the linked list having twin sum of 1 + 100000 = 100001.
    """
    """
    Hints: Use a stack for less code, still efficient
    """
    def pairSumLL(self, head: ListNode) -> int:
        return 0

    """
    707. Design Linked List
    Check MyLinkedList CLASS
    """

    """
    92. Reverse Linked List II
    Given the head of a singly linked list and two integers left and right where left <= right, 
    reverse the nodes of the list from position left to position right, and return the reversed list.
    Ex1: Input: head = [1,2,3,4,5], left = 2, right = 4, Output: [1,4,3,2,5]
    Ex2: Input: head = [5], left = 1, right = 1, Output: [5]
    """
    """
    Hints: reach left node with a pointer behind it, reverse from left to right, update pointers
    """
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        return None

    """
    622. Design Circular Queue
    Check MyCircularQueue CLASS
    """

    #################### Trees ############################### 10 - 45
    """
    - Binary Tree: DFS(Recursive)
    def dfs(root):
        if not root: return
        ans = 0
        # do logic
        dfs(root.left)
        dfs(root.right)
        return ans

    - Binary Tree: DFS (iterative)
    def dfs(root):
    stack, ans = [root], 0
    while stack:
        node = stack.pop()
        # do logic
        if node.left: stack.append(node.left)
        if node.right: stack.append(node.right)
    return ans

    - Binary Tree: BFS
    def fn(root):
    queue, ans = deque([root]), 0
    while queue:
        current_length = len(queue)
        # do logic for current level
        for _ in range(current_length):
            node = queue.popleft()
            # do logic
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return ans
    """

    """
    226. Invert Binary Tree
    Given the root of a binary tree, invert the tree and return the root
    Ex1: root = [4,2,7,1,3,6,9], Output: [4,7,2,9,6,3,1]
    Ex2: root = [2, 1, 3], Output: [2, 3, 1]
    """
    """
    Hints: One-liner use the constructor of the TreeNode
    """
    def invertTree(self, root: TreeNode) -> TreeNode:
        return None

    """
    104. Maximum Depth of Binary Tree
    Given the root of a binary tree, return its maximum depth.
    A binary tree's maximum depth is the number of nodes along 
    the longest path from the root node down to the farthest leaf node.
    Ex1: Input: root = [3,9,20,null,null,15,7], Output: 3
    """
    """
    Hints: get the max
    """
    def maxDepth(self, root: TreeNode) -> int:
        return 0

    """
    543. Diameter of Binary Tree
    Given the root of a binary tree, return the length of the diameter of the tree.
    The diameter of a binary tree is the length of the longest path 
    between any two nodes in a tree. This path may or may not pass through the root.
    The length of a path between two nodes is represented by the number of edges between them.
    Ex1: Input: root = [1,2,3,4,5], Output: 3
    Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
    Ex2: Input: root = [1,2], Output: 1
    """
    """
    Hints: dfs, non res, max, 1+max
    """
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        return 0

    """
    110. Balanced Binary Tree
    Given a binary tree, determine if it is height-balanced
    Ex1: Input: root = [3,9,20,null,null,15,7], Output: true
    """
    """
    Hints: get the height, abs(left-right)>1: -1
    """
    def isBalanced(self, root: TreeNode) -> bool:
        return False

    """
    100. Same Tree
    Given the roots of two binary trees p and q, write a function to check if they are the same or not.
    Two binary trees are considered the same if they are structurally identical, 
    and the nodes have the same value.
    Ex1: Input: p = [1,2,3], q = [1,2,3], Output: true
    Ex2: Input: p = [1,2], q = [1,null,2], Output: false
    """
    """
    Hints: check, check, return else return
    """
    def isSametree(self, p: TreeNode, q: TreeNode) -> bool:
        return False

    """
    572. Subtree of Another Tree
    Given the roots of two binary trees root and subRoot, return true if there is a subtree of root 
    with the same structure and node values of subRoot and false otherwise. A subtree of a binary 
    tree is a tree that consists of a node in tree and all of this node's descendants. 
    The tree tree could also be considered as a subtree of itself.
    Ex1: Input: root = [3,4,5,1,2], subRoot = [4,1,2], Output: true
    Ex2: Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2], Output: false
    """
    """
    Hints: use isSameTree, 
    """
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        return False

    """
    235. Lowest Common Ancestor of a Binary Search Tree
    Given a binary search tree (BST), find the lowest common ancestor (LCA) 
    node of two given nodes in the BST.
    According to the definition of LCA on Wikipedia:
    “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has 
    both p and q as descendants (where we allow a node to be a descendant of itself).”
    Ex1: Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8, Output: 6
    Explanation: The LCA of nodes 2 and 8 is 6.
    Ex2: Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4, Output: 2
    Explanation: The LCA of nodes 2 and 4 is 2, 
    since a node can be a descendant of itself according to the LCA definition.
    """
    """
    Hints: check the val of p and q then move based on that
    """
    def lowestCmmonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        return None

    """
    102. Binary Tree Level Order Traversal
    Given the root of a binary tree, return the level order traversal of its nodes' values. 
    (i.e., from left to right, level by level).
    Ex1: Input: root = [3,9,20,null,null,15,7], Output: [[3],[9,20],[15,7]]
    """
    """
    Hints: BFS bro
    """
    def levelOrder(self, root: TreeNode) -> list[list[int]]:
        return [[]]

    """
    199. Binary Tree Right Side View
    Given the root of a binary tree, imagine yourself standing on the right side of it, 
    return the values of the nodes you can see ordered from top to bottom.
    Ex1: Input: root = [1,2,3,null,5,null,4], Output: [1,3,4]
    Ex2: Input: root = [1,null,3], Output: [1,3]
    """
    def rightSideView(self, root: TreeNode) -> list[int]:
        return []

    """
    1448. Count Good Nodes in Binary Tree
    Given a binary tree root, a node X in the tree is named good if in the path 
    from root to X there are no nodes with a value greater than X.
    Return the number of good nodes in the binary tree.
    Ex1: Input: root = [3,1,4,3,null,1,5], Output: 4
    Explanation: Nodes in blue are good.
    Root Node (3) is always a good node.
    Node 4 -> (3,4) is the maximum value in the path starting from the root.
    Node 5 -> (3,4,5) is the maximum value in the path
    Node 3 -> (3,1,3) is the maximum value in the path.
    Ex2: Input: root = [3,3,null,4,2], Output: 3
    Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.
    """
    """
    Hints: dfs(root, maxVal)
    """
    def goodNode(self, root: TreeNode) -> int:
        return 0

    """
    98. Validate Binary Search Tree
    Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    A valid BST is defined as follows:
    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.
    Ex1: Input: root = [2,1,3], Output: true
    Ex2: Input: root = [5,1,4,null,null,3,6], Output: false
    Explanation: The root node's value is 5 but its right child's value is 4.
    """
    """
    Hints: vaild(root, left, right)
    """
    def isValidBST(self, root: TreeNode) -> bool:
        return False

    """
    230. Kth Smallest Element in a BST
    Given the root of a binary search tree, and an integer k, 
    return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
    Ex1: Input: root = [3,1,4,null,2], k = 1, Output: 1
    """
    """
    Hints: BFS, cur, q
    """
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        return 0

    """
    105. Construct Binary Tree from Preorder and Inorder Traversal
    Given two integer arrays preorder and inorder where preorder is 
    the preorder traversal of a binary tree and inorder is 
    the inorder traversal of the same tree, construct and return the binary tree.
    Ex1: Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7], Output: [3,9,20,null,null,15,7]
    """
    """
    Hints:
    """
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        return None

    """
    124. Binary Tree Maximum Path Sum - HARD
    A path in a binary tree is a sequence of nodes where each pair of adjacent nodes 
    in the sequence has an edge connecting them. 
    A node can only appear in the sequence at most once. 
    Note that the path does not need to pass through the root.
    The path sum of a path is the sum of the node's values in the path.
    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    Ex1: Input: root = [1,2,3], Output: 6
    Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
    Ex2: Input: root = [-10,9,20,null,null,15,7], Output: 42
    Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
    """
    """
    Hints:
    """
    def maxPathSum(self, root: TreeNode) -> int:
        return 0

    """
    297. Serialize and Deserialize Binary Tree  - HARD
    Serialization is the process of converting a data structure or object into a sequence of 
    bits so that it can be stored in a file or memory buffer, or transmitted across a network connection 
    link to be reconstructed later in the same or another computer environment.
    Design an algorithm to serialize and deserialize a binary tree. There is no restriction on 
    how your serialization/deserialization algorithm should work. 
    You just need to ensure that a binary tree can be serialized to a string and 
    this string can be deserialized to the original tree structure.
    Clarification: The input/output format is the same as how LeetCode serializes a binary tree. 
    You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
    Ex1: Input: root = [1,2,3,null,null,4,5], Output: [1,2,3,null,null,4,5]
    *** PREORDER 
    """
    """
    Hints:
    """
    def serialize(self, root) -> str: return ""
    def deserialize(self, data) -> TreeNode: return None


    # new Picks from 300 list
    """
    144. Binary Tree Preorder Traversal
    Given the root of a binary tree, return the preorder traversal of its nodes' values.
    Ex1: Input: root = [1,null,2,3], Output: [1,2,3]
    Ex2: Input: root = [], Output: []
    Ex3: Input: root = [1], Output: [1]
    """
    def preOrderTraversal(self, root: TreeNode) -> list[int]:
        return []

    """
    94. Binary Tree Inorder Traversal
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
    Ex1: Input: root = [1,null,2,3], Output: [1,3,2]
    Ex2: Input: root = [], Output: []
    Ex3: Input: root = [1], Output: [1]
    """
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        return []

    """
    145. Binary Tree PostOrder Traversal
    Given the root of a binary tree, return the postorder traversal of its nodes' values.
    Ex1: Input: root = [1,null,2,3], Output: [3,2,1]
    Ex2: Input: root = [], Output: []
    Ex3: Input: root = [1], Output: [1]
    """
    def postOrderTraversal(self, root: TreeNode) -> list[int]:
        return []

    """
    108. Convert Sorted Array to Binary Search Tree
    Given an integer array nums where the elements are sorted in ascending order, convert it to a 
    height-balanced binary search tree.
    Ex1: Input: nums = [-10,-3,0,5,9], Output: [0,-3,9,-10,null,5]
        Explanation: [0,-10,5,null,-3,null,9] is also accepted:
    """
    """
    Hints: Play on the mid, then recurse using the mid
    """
    def sortedArrayToBST(self, nums: list[int]) -> TreeNode:
        return None

    """
    701. Insert into a Binary Search Tree
    You are given the root node of a binary search tree (BST) and a value to insert into the tree. 
    Return the root node of the BST after the insertion. 
    It is guaranteed that the new value does not exist in the original BST.
    Notice that there may exist multiple valid ways for the insertion, 
    as long as the tree remains a BST after insertion. You can return any of them.
    Ex1: Input: root = [4,2,7,1,3], val = 5, Output: [4,2,7,1,3,5]
    Ex2: Input: root = [40,20,60,10,30,50,70], val = 25, Output: [40,20,60,10,30,50,70,null,null,25]
    Ex3: Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5, Output: [4,2,7,1,3,5]
    """
    """
    Hints: recursive
    """
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        return None

    """
    450. Delete node in a BST
    Given a root node reference of a BST and a key, delete the node with the given key in the BST. 
    Return the root node reference (possibly updated) of the BST.
    Basically, the deletion can be divided into two stages: Search for a node to remove.
    If the node is found, delete the node.
    Ex1: Input: root = [5,3,6,2,4,null,7], key = 3, Output: [5,4,6,2,null,null,7]
        Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
        One valid answer is [5,4,6,2,null,null,7], shown in the above BST.
        Please notice that another valid answer is [5,2,6,null,4,null,7] and it's also accepted.
    Ex2: Input: root = [5,3,6,2,4,null,7], key = 0, Output: [5,3,6,2,4,null,7]
        Explanation: The tree does not contain a node with value = 0.
    NEED to build the successor and predessesor methods
    """
    """
    Hints: search for the key while recursing, check children, then return successor
    """
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return None
        if root.val < key:  root.right = self.deleteNode(root.right, key)
        elif root.val > key: root.left = self.deleteNode(root.left, key)
        else:
            if not root.left and not root.right: root = None
            if not root.left: return root.right
            if not root.right: return root.left
            if root.left and root.right:
                node = root.right
                while node.left: node = node.left
                root.val = node.val
                root.right = self.deleteNode(root.right, root.val)
        return root

    """
    617. Merge Two Binary Trees
    You are given two binary trees root1 and root2.
    Imagine that when you put one of them to cover the other, 
    some nodes of the two trees are overlapped while the others are not. 
    You need to merge the two trees into a new binary tree. 
    The merge rule is that if two nodes overlap, 
    then sum node values up as the new value of the merged node. 
    Otherwise, the NOT null node will be used as the node of the new tree.
    Return the merged tree.
    Note: The merging process must start from the root nodes of both trees.
    Ex1: Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7], Output: [3,4,5,5,4,null,7]
    Ex2: Input: root1 = [1], root2 = [1,2], Output: [2,2]
    """
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        return None

    """
    652. Find Duplicate Subtrees
    Given the root of a binary tree, return all duplicate subtrees.
    For each kind of duplicate subtrees, you only need to return the root node of any one of them.
    Two trees are duplicate if they have the same structure with the same node values.
    Ex1: Input: root = [1,2,3,4,null,2,4,null,null,4], Output: [[2,4],[4]]
    Ex2: Input: root = [2,1,1], Output: [[1]]
    Ex3: Input: root = [2,2,2,3,null,3,null], Output: [[2,3],[3]]
    """
    """
    Hints: need a map
    """
    def findDuplicateSubtrees(self, root: TreeNode) -> list[TreeNode]:
        return None

    #################### Tries ############################### 2 - 3
    """
    - Build a Trie
    # note: using a class is only necessary if you want to store data at each node.
    # otherwise, you can implement a trie using only hash maps.
    class TrieNode:
        def __init__(self):
            # you can store data at nodes if you wish
            self.data = None
            self.children = {}
    def fn(words):
        root = TrieNode()
        for word in words:
            curr = root
            for c in word:
                if c not in curr.children:
                    curr.children[c] = TrieNode()
                curr = curr.children[c]
            # at this point, you have a full word at curr
            # you can perform more logic here to give curr an attribute if you want
        return root
    """

    """
    208. Implement Trie (Prefix Tree) - TRIE CLASS
    """

    """
    211. Design Add and Search Words in Data Structure - WordDictionary CLASS
    """

    """
    212. Word Search II - HARD
    Given an m x n board of characters and a list of strings words, return all words on the board.
    Each word must be constructed from letters of sequentially adjacent cells, 
    where adjacent cells are horizontally or vertically neighboring. 
    The same letter cell may not be used more than once in a word.
    Ex1: Input: board = [["o","a","a","n"],
                         ["e","t","a","e"],
                         ["i","h","k","r"],
                         ["i","f","l","v"]], words = ["oath","pea","eat","rain"] - Output: ["eat","oath"]
    Ex2: Input: board = [["a","b"],["c","d"]], words = ["abcb"] - Output: []
    """
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        return None

    #################### Heap / Priority Queue ############### - 18
    """
    - Find top K elements with Heap
    def fn(arr, k):
        heap = []
        for num in arr:
            # do some logic to push onto heap according to problem's criteria
            heapq.heappush(heap, (CRITERIA, num))
            if len(heap) > k: heapq.heappop(heap)
        return [num for num in heap]
    """

    """
    703. kth largest Element in a Stream - kthLargestElement CLASS
    """

    """
    1046. Last Stone Weight
    You are given an array of integers stones where stones[i] is the weight of the ith stone.
    We are playing a game with the stones. On each turn, 
    we choose the heaviest two stones and smash them together. 
    Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:
    If x == y, both stones are destroyed, and
    If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
    At the end of the game, there is at most one stone left.
    Return the weight of the last remaining stone. If there are no stones left, return 0.
    Ex1: Input: stones = [2,7,4,1,8,1], Output: 1
    Explanation: 
        We combine 7 and 8 to get 1 so the array converts to [2,4,1,1,1] then,
        we combine 2 and 4 to get 2 so the array converts to [2,1,1,1] then,
        we combine 2 and 1 to get 1 so the array converts to [1,1,1] then,
        we combine 1 and 1 to get 0 so the array converts to [1] then that's the value of the last stone.
    """
    """
    Hints: Use MAX heap
    """
    def lastStoneWeight(self, stones: list[int]) -> int:
        return 0

    """
    973. K Closest Points to the Origin
    Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, 
    return the k closest points to the origin (0, 0).
    The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).
    You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).
    Ex1: Input: points = [[1,3],[-2,2]], k = 1, Output: [[-2,2]]
    Explanation:
        The distance between (1, 3) and the origin is sqrt(10).
        The distance between (-2, 2) and the origin is sqrt(8).
        Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
        We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
    Ex2: Input: points = [[3,3],[5,-1],[-2,4]], k = 2, Output: [[3,3],[-2,4]]
    Explanation: The answer [[-2,4],[3,3]] would also be accepted.
    """
    """
    Hints: min Heap, calc dis, use pushpop if len > k
    """
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        return None

    """
    215. Kth Largest Element in an Array
    Given an integer array nums and an integer k, return the kth largest element in the array.
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    Can you solve it without sorting?
    Ex1: Input: nums = [3,2,1,5,6,4], k = 2, Output: 5
    Ex2: Input: nums = [3,2,3,1,2,4,5,5,6], k = 4, Output: 4
    """
    """
    Hints: 
    """
    def findKthLargest(self, nums: list[int], k: int) -> int:
        return 0

    """
    621. Task Scheduler
    Given a characters array tasks, representing the tasks a CPU needs to do, 
    where each letter represents a different task. Tasks could be done in any order. 
    Each task is done in one unit of time. 
    For each unit of time, the CPU could complete either one task or just be idle.
    However, there is a non-negative integer n that represents the cooldown period 
    between two same tasks (the same letter in the array), 
    that is that there must be at least n units of time between any two same tasks.
    Return the least number of units of times that the CPU will take to finish all the given tasks.
    Ex1: Input: tasks = ["A","A","A","B","B","B"], n = 2, Output: 8
    Explanation: 
        A -> B -> idle -> A -> B -> idle -> A -> B
        There is at least 2 units of time between any two same tasks.
    Ex2: Input: tasks = ["A","A","A","B","B","B"], n = 0, Output: 6
    Explanation: On this case any permutation of size 6 would work since n = 0.
        ["A","A","A","B","B","B"]
        ["A","B","A","B","A","B"]
        ["B","B","B","A","A","A"]
        ...
        And so on.
    Ex3: Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2, Output: 16
    Explanation: 
        One possible solution is
        A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
    """
    """
    Hints: use Counter, get the max val, and the count of it
    """
    def leastInterval(self, tasks: list[str], n: int) -> int:
        return 0

    """
    355. Design Twitter - Twitter CLASS
    """

    """
    295. Find Median from Data Stream - MedianFinder CLASS - HARD
    """

    # new Picks from 300 list

    """
    1985. Find the Kth largest Integer in the Array
    You are given an array of strings nums and an integer k. 
    Each string in nums represents an integer without leading zeros.
    Return the string that represents the kth largest integer in nums.
    Note: Duplicate numbers should be counted distinctly. 
    For example, if nums is ["1","2","2"], "2" is the first largest integer, 
    "2" is the second-largest integer, and "1" is the third-largest integer.
    Ex1: Input: nums = ["3","6","7","10"], k = 4, Output: "3"
    Explanation:
        The numbers in nums sorted in non-decreasing order are ["3","6","7","10"].
        The 4th largest integer in nums is "3".
    Ex2: Input: nums = ["2","21","12","1"], k = 3, Output: "2"
    Explanation:
        The numbers in nums sorted in non-decreasing order are ["1","2","12","21"].
        The 3rd largest integer in nums is "2".
    Ex3: Input: nums = ["0","0"], k = 2, Output: "0"
    Explanation:
        The numbers in nums sorted in non-decreasing order are ["0","0"].
        The 2nd largest integer in nums is "0".
    """
    """
    Hints: You can use SortedList One liner solution
    """
    def kthLargestNumber(self, nums: list[str], k: int) -> str:
        return ""

    """
    1834. Single Threaded CPU
    You are given n tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, 
    where tasks[i] = [enqueueTimei, processingTimei] means that the ith task will be available 
    to process at enqueueTimei and will take processingTimei to finish processing.
    You have a single-threaded CPU that can process at most one task at a time and 
    will act in the following way:
    If the CPU is idle and there are no available tasks to process, the CPU remains idle.
    If the CPU is idle and there are available tasks, the CPU will choose the one 
    with the shortest processing time. If multiple tasks have the same shortest processing time, 
    it will choose the task with the smallest index.
    Once a task is started, the CPU will process the entire task without stopping.
    The CPU can finish a task then start a new one instantly.
    Return the order in which the CPU will process the tasks.
    Ex1: Input: tasks = [[1,2],[2,4],[3,2],[4,1]], Output: [0,2,3,1]
    Explanation: The events go as follows: 
    - At time = 1, task 0 is available to process. Available tasks = {0}.
    - Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
    - At time = 2, task 1 is available to process. Available tasks = {1}.
    - At time = 3, task 2 is available to process. Available tasks = {1, 2}.
    - Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. 
            Available tasks = {1}.
    - At time = 4, task 3 is available to process. Available tasks = {1, 3}.
    - At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. 
            Available tasks = {1}.
    - At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
    - At time = 10, the CPU finishes task 1 and becomes idle.
    Ex2: Input: tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]], Output: [4,3,2,0,1]
    Explanation: The events go as follows:
    - At time = 7, all the tasks become available. Available tasks = {0,1,2,3,4}.
    - Also at time = 7, the idle CPU starts processing task 4. Available tasks = {0,1,2,3}.
    - At time = 9, the CPU finishes task 4 and starts processing task 3. Available tasks = {0,1,2}.
    - At time = 13, the CPU finishes task 3 and starts processing task 2. Available tasks = {0,1}.
    - At time = 18, the CPU finishes task 2 and starts processing task 0. Available tasks = {1}.
    - At time = 28, the CPU finishes task 0 and starts processing task 1. Available tasks = {}.
    - At time = 40, the CPU finishes task 1 and becomes idle.
    """
    """
    Hints: sort task and add index to it, build min heap and check prevTim with curr time then get the max time
    """
    def getOrder(self, tasks: list[list[int]]) -> list[int]:
        return None

    """
    2542. Maximum Subsequence Score
    You are given two 0-indexed integer arrays nums1 and nums2 of equal length n and a positive integer k. 
    You must choose a subsequence of indices from nums1 of length k.
    For chosen indices i0, i1, ..., ik - 1, your score is defined as:
    The sum of the selected elements from nums1 multiplied with the minimum 
    of the selected elements from nums2.
    It can defined simply as: (nums1[i0] + nums1[i1] +...+ nums1[ik - 1]) * 
                                min(nums2[i0] , nums2[i1], ... ,nums2[ik - 1]).
    Return the maximum possible score.
    A subsequence of indices of an array is a set that can be derived from the 
    set {0, 1, ..., n-1} by deleting some or no elements.
    Ex1: Input: nums1 = [1,3,3,2], nums2 = [2,1,3,4], k = 3, Output: 12
    Explanation: The four possible subsequence scores are:
    - We choose the indices 0, 1, and 2 with score = (1+3+3) * min(2,1,3) = 7.
    - We choose the indices 0, 1, and 3 with score = (1+3+2) * min(2,1,4) = 6. 
    - We choose the indices 0, 2, and 3 with score = (1+3+2) * min(2,3,4) = 12. 
    - We choose the indices 1, 2, and 3 with score = (3+3+2) * min(1,3,4) = 8.
    Therefore, we return the max score, which is 12.
    Ex2: Input: nums1 = [4,2,3,1,1], nums2 = [7,5,10,9,6], k = 1, Output: 30
    Explanation: 
    Choosing index 2 is optimal: nums1[2] * nums2[2] = 3 * 10 = 30 is the maximum possible score.
    """
    """
    Hints: zip, sort, heapify, iterate
    """
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

    """
    1882. Process Tasks Using Servers
    You are given two 0-indexed integer arrays servers and tasks of lengths n and m respectively. 
    servers[i] is the weight of the ith server, and tasks[j] is the time needed to process 
    the jth task in seconds. Tasks are assigned to the servers using a task queue. 
    Initially, all servers are free, and the queue is empty.
    At second j, the jth task is inserted into the queue (starting with 
    the 0th task being inserted at second 0). 
    As long as there are free servers and the queue is not empty, 
    the task in the front of the queue will be assigned to a free server with the smallest weight, 
    and in case of a tie, it is assigned to a free server with the smallest index.
    If there are no free servers and the queue is not empty, we wait until a server 
    becomes free and immediately assign the next task. 
    If multiple servers become free at the same time, then multiple tasks from the queue will 
    be assigned in order of insertion following the weight and index priorities above.
    A server that is assigned task j at second t will be free again at second t + tasks[j].
    Build an array ans of length m, where ans[j] is the index of the server 
    the jth task will be assigned to. Return the array ans.
    Ex1: Input: servers = [3,3,2], tasks = [1,2,3,2,1,2], Output: [2,2,0,2,1,2]
    Explanation: Events in chronological order go as follows:
    - At second 0, task 0 is added and processed using server 2 until second 1.
    - At second 1, server 2 becomes free. Task 1 is added and processed using server 2 until second 3.
    - At second 2, task 2 is added and processed using server 0 until second 5.
    - At second 3, server 2 becomes free. Task 3 is added and processed using server 2 until second 5.
    - At second 4, task 4 is added and processed using server 1 until second 5.
    - At second 5, all servers become free. Task 5 is added and processed using server 2 until second 7.
    Ex2: Input: servers = [5,1,4,3,2], tasks = [2,1,2,4,5,2,1], Output: [1,4,1,4,1,3,2]
    Explanation: Events in chronological order go as follows: 
    - At second 0, task 0 is added and processed using server 1 until second 2.
    - At second 1, task 1 is added and processed using server 4 until second 2.
    - At second 2, servers 1 and 4 become free. Task 2 is added and processed using server 1 until second 4. 
    - At second 3, task 3 is added and processed using server 4 until second 7.
    - At second 4, server 1 becomes free. Task 4 is added and processed using server 1 until second 9. 
    - At second 5, task 5 is added and processed using server 3 until second 7.
    - At second 6, task 6 is added and processed using server 2 until second 7.
    """
    """
    Hints: use a heap
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


    #################### Backtracking ########################10 - 18
    """
    - Backtracking
    def backtrack(curr, OTHER_ARGUMENTS...):
        if (BASE_CASE):
            # modify the answer
            return
        ans = 0
        for (ITERATE_OVER_INPUT):
            # modify the current state
            ans += backtrack(curr, OTHER_ARGUMENTS...)
            # undo the modification of the current state
        return ans
    """

    """
    78. Subsets
    Given an integer array nums of unique elements, return all possible subsets (the power set).
    The solution set must not contain duplicate subsets. Return the solution in any order.
    Ex1: Input: nums = [1,2,3], Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    """
    """
    Hints: unique -> DFS, or use itertools
    """
    def subsets(self, nums: list[int]) -> list[list[int]]:
        return None

    """
    90. Subset II
    Given an integer array nums that may contain duplicates, return all possible subsets(the power set).
    The solution set must not contain duplicate subsets. Return the solution in any order. 
    Ex1: Input: nums = [1,2,2], Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
    Ex2: Input: nums = [0], Output: [[],[0]]
    """
    """
    Hints: duplicates -> Either check the elements after you pop or choose not to include the subset if its in the list
    """
    def subsetsIIWithDup(self, nums: list[int]) -> list[list[int]]:
        return None

    """
    46. Permutations
    Given an array nums of distinct integers, return all the possible permutations. 
    You can return the answer in any order.
    Ex1: Input: nums = [1,2,3], Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    Ex2: Input: nums = [0,1], Output: [[0,1],[1,0]]
    """
    """
    Hint: DFS
    """
    def permute(self, nums: list[int]) -> list[list[int]]:
        return None

    """
    131. Palindrome Partitioning
    Given a string s, partition s such that every substring of the partition is a palindrome
    Return all possible palindrome partitioning of s.
    Ex1: Input: s = "aab", Output: [["a","a","b"],["aa","b"]]
    """
    """
    Hints: build isPali
    """
    def palindromePartitioning(self, s: str) -> list[list[str]]:
        return []

    """
    17. Letter Combinations of a phone Number
    Given a string containing digits from 2-9 inclusive, 
    return all possible letter combinations that the number could represent. 
    Return the answer in any order.
    A mapping of digits to letters (just like on the telephone buttons) is given below. 
    Note that 1 does not map to any letters.
    Ex1: Input: digits = "23", Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    Ex2: Input: digits = "2", Output: ["a","b","c"]
    """
    """
    Hints:
    """
    def letterCombinations(self, digits: str) -> list[str]:
        return []

    """
    39. Combination Sum
    Given an array of distinct integers candidates and a target integer target, 
    return a list of all unique combinations of candidates where the chosen numbers sum to target. 
    You may return the combinations in any order.
    The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
    frequency of at least one of the chosen numbers is different.
    The test cases are generated such that the number of unique combinations that sum up to target 
    is less than 150 combinations for the given input.
    Ex1: Input: candidates = [2,3,6,7], target = 7, Output: [[2,2,3],[7]]
    Explanation:
        2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
        7 is a candidate, and 7 = 7. These are the only two combinations.
    Ex2: Input: candidates = [2,3,5], target = 8, Output: [[2,2,2,2],[2,3,3],[3,5]]
    """
    """
    Hints: DFS
    """
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        return None

    """
    40. Combination Sum II
    Given a collection of candidate numbers (candidates) and a target number (target), 
    find all unique combinations in candidates where the candidate numbers sum to target.
    Each number in candidates may only be used once in the combination.
    Note: The solution set must not contain duplicate combinations.
    Ex1: Input: candidates = [10,1,2,7,6,1,5], target = 8, Output: [[1,1,6],[1,2,5],[1,7],[2,6]]
    Ex2: Input: candidates = [2,5,2,1,2], target = 5, Output:[[1,2,2],[5]]
    """
    """
    Hints: DFS
    """
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        return None

    """
    79. Word Search I
    Given an m x n grid of characters board and a string word, 
    return true if word exists in the grid.
    The word can be constructed from letters of sequentially adjacent cells, 
    where adjacent cells are horizontally or vertically neighboring. 
    The same letter cell may not be used more than once.
    Ex1: Input: board = [["A","B","C","E"],
                         ["S","F","C","S"],
                         ["A","D","E","E"]], word = "ABCCED", Output: true
    Ex2: Input: board = [["A","B","C","E"],
                         ["S","F","C","S"],
                         ["A","D","E","E"]], word = "SEE", Output: true
    Ex3: Input: board = [["A","B","C","E"],
                         ["S","F","C","S"],
                         ["A","D","E","E"]], word = "ABCB", Output: false
         O(n * m * 4^n)
    """
    """
    Hints: DFS, then 2-for loops
    """
    def wordSearchExist(self, board: list[list[str]], word: str) -> bool:
        return False

    """
    51. N-Queens - HARD
    The n-queens puzzle is the problem of placing n queens on an n x n chessboard 
    such that no two queens attack each other.
    Given an integer n, return all distinct solutions to the n-queens puzzle. 
    You may return the answer in any order.
    Each solution contains a distinct board configuration of the n-queens' placement, 
    where 'Q' and '.' both indicate a queen and an empty space, respectively.
    Ex1: Input: n = 4
    Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
    """
    """
    Hints:
    """
    def solveNQueens(self, n: int) -> list[list[str]]:
        return []

    # new Picks from 300 list

    """
    77. Combinations
    Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].
    You may return the answer in any order.
    Ex1: Input: n = 4, k = 2, Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    Explanation: There are 4 choose 2 = 6 total combinations.
    Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.
    Ex2: Input: n = 1, k = 1, Output: [[1]]
    Explanation: There is 1 choose 1 = 1 total combination.
    """
    """
    Hints: DFS with a for loop
    """
    def combine(self, n: int, k: int) -> list[list[int]]:
        return None

    """
    47. Permutations II
    Given a collection of numbers, nums, that might contain duplicates, 
    return all possible unique permutations in any order.
    Ex1: Input: nums = [1,1,2], Output:[[1,1,2], [1,2,1], [2,1,1]]
    Ex2: Input: nums = [1,2,3], Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    """
    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        res, count, perm = [], Counter(nums), []
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

    """
    93. Restore IP Addresses
    A valid IP address consists of exactly four integers separated by single dots. 
    Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.
    For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, 
    but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.
    Given a string s containing only digits, return all possible valid IP addresses 
    that can be formed by inserting dots into s. 
    You are not allowed to reorder or remove any digits in s. 
    You may return the valid IP addresses in any order.
    Ex1: Input: s = "25525511135", Output: ["255.255.11.135","255.255.111.35"]
    Ex2: Input: s = "0000", Output: ["0.0.0.0"]
    Ex3: Input: s = "101023", Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
    """
    def restoreIPAddresses(self, s: str) -> list[str]:
        return ['']

    """
    473. Matchsticks to square
    You are given an integer array matchsticks where matchsticks[i] is the length of the ith matchstick. 
    You want to use all the matchsticks to make one square. You should not break any stick, 
    but you can link them up, and each matchstick must be used exactly one time.
    Return true if you can make this square and false otherwise.
    Ex1: Input: matchsticks = [1,1,2,2,2], Output: true
    Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.
    Ex2: Input: matchsticks = [3,3,3,3,4], Output: false
    Explanation: You cannot find a way to form a square with all the matchsticks.
    """
    def makesquare(self, matchsticks: list[int]) -> bool:
        return False

    #################### Graphs ############################## - 38
    """
    - Graph: DFS (recursive)
    def fn(graph):
        def dfs(node):
            ans = 0
            # do some logic
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    ans += dfs(neighbor)
            return ans
        seen = {START_NODE}
    return dfs(START_NODE)
    
    - Graph: DFS (iterative)
    def fn(graph):
        stack = [START_NODE]
        seen = {START_NODE}
        ans = 0
        while stack:
            node = stack.pop()
            # do some logic
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return ans
        
    - Graph: BFS
    def fn(graph):
    queue, seen, ans = deque([START_NODE]), {START_NODE}, 0
    while queue:
        node = queue.popleft()
        # do some logic
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return ans
    """

    """
    200. Number of Island
    Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
    return the number of islands.
    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
    You may assume all four edges of the grid are all surrounded by water.
    Ex1: Input: grid = [["1","1","1","1","0"],["1","1","0","1","0"],
                        ["1","1","0","0","0"],["0","0","0","0","0"]], Output: 1
    Ex2: Input: grid = [["1","1","0","0","0"],["1","1","0","0","0"],
                        ["0","0","1","0","0"],["0","0","0","1","1"]], Output: 3
    """
    """
    Hints: for in dfs with dirs
    """
    def numIslands(self, grid: list[list[str]]) -> int:
        return 0

    """
    133. Clone Graph
    Given a reference of a node in a connected undirected graph.
    Return a deep copy (clone) of the graph.
    Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
    For simplicity, each node's value is the same as the node's index (1-indexed). 
    For example, the first node with val == 1, the second node with val == 2, and so on. 
    The graph is represented in the test case using an adjacency list.
    An adjacency list is a collection of unordered lists used to represent a finite graph. 
    Each list describes the set of neighbors of a node in the graph.
    The given node will always be the first node with val = 1. 
    You must return the copy of the given node as a reference to the cloned graph.
    Ex1: Input: adjList = [[2,4],[1,3],[2,4],[1,3]], Output: [[2,4],[1,3],[2,4],[1,3]]
    Explanation: There are 4 nodes in the graph.
        1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
        2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
        3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
        4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
    Ex2: Input: adjList = [[]], Output: [[]]
    Explanation: Note that the input contains one empty list. 
        The graph consists of only one node with val = 1 and it does not have any neighbors.
    Ex3: Input: adjList = [], Output: []
    Explanation: This an empty graph, it does not have any nodes.
    """
    """
    Hints: for in dfs for nei
    """
    def cloneGraph(self, node: Node) -> Node:
        return None

    """
    695. Max Area of Island
    You are given an m x n binary matrix grid. 
    An island is a group of 1's (representing land) connected 
    4-directionally (horizontal or vertical.) 
    You may assume all four edges of the grid are surrounded by water.
    The area of an island is the number of cells with a value 1 in the island.
    Return the maximum area of an island in grid. If there is no island, return 0.
    Ex1: Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,1,1,0,0,0],
                        [0,1,1,0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,1,1,0,0,1,0,1,0,0],
                        [0,1,0,0,1,1,0,0,1,1,1,0,0],
                        [0,0,0,0,0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,1,1,0,0,0,0]], Output: 6
    Explanation: The answer is not 11, because the island must be connected 4-directionally.
    """
    """
    Hints: DFS, check > >= return 1 + dfssss
    """
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        return 0

    """
    417. Pacific Atlantic Water Flow
    There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. 
    The Pacific Ocean touches the island's left and top edges, 
    and the Atlantic Ocean touches the island's right and bottom edges.
    The island is partitioned into a grid of square cells. 
    You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level 
    of the cell at coordinate (r, c).
    The island receives a lot of rain, and the rain water can flow to neighboring cells 
    directly north, south, east, and west if the neighboring cell's height is less than or equal to 
    the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.
    Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water 
    can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
    Ex1: Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
    Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
    [0,4]: [0,4] -> Pacific Ocean 
           [0,4] -> Atlantic Ocean
    [1,3]: [1,3] -> [0,3] -> Pacific Ocean 
           [1,3] -> [1,4] -> Atlantic Ocean
    [1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
           [1,4] -> Atlantic Ocean
    [2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
           [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
    [3,0]: [3,0] -> Pacific Ocean 
           [3,0] -> [4,0] -> Atlantic Ocean
    [3,1]: [3,1] -> [3,0] -> Pacific Ocean 
           [3,1] -> [4,1] -> Atlantic Ocean
    [4,0]: [4,0] -> Pacific Ocean 
           [4,0] -> Atlantic Ocean
    Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.
    Ex2: Input: heights = [[1]], Output: [[0,0]]
    Explanation: The water can flow from the only cell to the Pacific and Atlantic oceans.
    """
    """
    Hints: DFS, then 3-fors
    """
    def pacificAtlantic(self, heights: list[list[int]]) -> list[list[int]]:
        return [[]]

    """
    130. Surrounded Regions
    Given an m x n matrix board containing 'X' and 'O', capture all regions 
    that are 4-directionally surrounded by 'X'.
    A region is captured by flipping all 'O's into 'X's in that surrounded region.
    Ex1: Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    Explanation: Notice that an 'O' should not be flipped if:
        - It is on the border, or
        - It is adjacent to an 'O' that should not be flipped.
        The bottom 'O' is on the border, so it is not flipped.
        The other three 'O' form a surrounded region, so they are flipped.
    Ex2: Input: board = [["X"]], Output: [["X"]]
    """
    """
    Hints: DFS, 3-nested fors. O->T, O->X, T->O
    """
    def solveSurroundedRegions(self, board: list[list[str]]) -> None:
        return None

    # new Picks from 300 list

    """
    463. Island Perimeter
    You are given row x col grid representing a map where grid[i][j] = 1 represents land 
    and grid[i][j] = 0 represents water.
    Grid cells are connected horizontally/vertically (not diagonally). 
    The grid is completely surrounded by water, and there is exactly one island 
    (i.e., one or more connected land cells).
    The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. 
    One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. 
    Determine the perimeter of the island.
    Ex1: Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]], Output: 16
        Explanation: The perimeter is the 16 yellow stripes in the image above.
    Ex2: Input: grid = [[1]], Output: 4
    Ex3: Input: grid = [[1,0]], Output: 4
    """
    """
    Hints: DFS
    """
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        return 0

    """
    953. Verifying an Alien Dictionary
    In an alien language, surprisingly, they also use English lowercase letters, 
    but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.
    Given a sequence of words written in the alien language, and the order of the alphabet, 
    return true if and only if the given words are sorted lexicographically in this alien language.
    Ex1: Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz", Output: true
    Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
    Ex2: Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz", Output: false
    Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], 
    hence the sequence is unsorted.
    Ex3: Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz", Output: false
    Explanation: The first three characters "app" match, and the second string is shorter (in size.) 
    According to lexicographical rules "apple" > "app", because 'l' > '∅', 
    where '∅' is defined as the blank character which is less than any other character (More info).
    """
    """
    Hints: DFS
    """
    def isAlienSorted(self, words: list[str], order: str) -> bool:
        return False

    """
    1905. Count Sub Islands
    You are given two m x n binary matrices grid1 and grid2 containing only 0's (representing water) 
    and 1's (representing land). An island is a group of 1's connected 4-directionally (horizontal or vertical). 
    Any cells outside of the grid are considered water cells.
    An island in grid2 is considered a sub-island if there is an island in 
    grid1 that contains all the cells that make up this island in grid2.
    Return the number of islands in grid2 that are considered sub-islands.
    Ex1: Input: grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]], 
                grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]], Output: 3
    Explanation: In the picture above, the grid on the left is grid1 and the grid on the right is grid2.
    The 1s colored red in grid2 are those considered to be part of a sub-island. There are three sub-islands.
    Ex2: Input: grid1 = [[1,0,1,0,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,0,1,0,1]], 
                grid2 = [[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[1,0,0,0,1]], Output: 2 
    Explanation: In the picture above, the grid on the left is grid1 and the grid on the right is grid2.
    The 1s colored red in grid2 are those considered to be part of a sub-island. There are two sub-islands.
    """
    """
    Hints: DFS
    """
    def countSubIslands(self, grid1: list[list[int]], grid2: list[list[int]]) -> int:
        return 0

    """
    1466. Reorder Routes to Make All Paths leads to the city Zero
    There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way to travel 
    between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction 
    because they are too narrow. Roads are represented by connections where connections[i] = [ai, bi] 
    represents a road from city ai to city bi. This year, there will be a big event in the capital (city 0), 
    and many people want to travel to this city. Your task consists of reorienting some roads such that each 
    city can visit the city 0. Return the minimum number of edges changed.
    It's guaranteed that each city can reach city 0 after reorder.
    Ex1: Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]], Output: 3
    Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).
    Ex2: Input: n = 5, connections = [[1,0],[1,2],[3,2],[3,4]], Output: 2
    Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).
    Ex3: Input: n = 3, connections = [[1,0],[2,0]], Output: 0
    """
    def minReorder(self, n: int, connections: list[list[int]]) -> int:
        return 0

    """
    752. Open the Lock
    You have a lock in front of you with 4 circular wheels. 
    Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. 
    The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. 
    Each move consists of turning one wheel one slot.
    The lock initially starts at '0000', a string representing the state of the 4 wheels.
    You are given a list of deadends dead ends, meaning if the lock displays any of these codes, 
    the wheels of the lock will stop turning and you will be unable to open it.
    Given a target representing the value of the wheels that will unlock the lock, 
    return the minimum total number of turns required to open the lock, or -1 if it is impossible.
    Ex1: Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202", Output: 6
    Explanation: 
        A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
        Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
        because the wheels of the lock become stuck after the display becomes the dead end "0102".
    Ex2: Input: deadends = ["8888"], target = "0009", Output: 1
        Explanation: We can turn the last wheel in reverse to move from "0000" -> "0009".
    Ex3: Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888", Output: -1
    Explanation: We cannot reach the target without getting stuck.
    """
    def openLock(self, deadends: list[str], target: str) -> int:
        return 0

    #################### Advance Graphs ###################### - 10
    """
    - Dijkstra's Algorithm
    distances, heap = [inf] * n, [(0, source)]
    distances[source] = 0
    while heap:
        curr_dist, node = heappop(heap)
        if curr_dist > distances[node]: continue
        for nei, weight in graph[node]:
            dist = curr_dist + weight
            if dist < distances[nei]:
                distances[nei] = dist
                heappush(heap, (dist, nei))
    """

    """
    1584. Min Cost to Connect All Points
    You are given an array points representing integer coordinates of 
    some points on a 2D-plane, where points[i] = [xi, yi].
    The cost of connecting two points [xi, yi] and [xj, yj] is the 
    manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.
    Return the minimum cost to make all points connected. 
    All points are connected if there is exactly one simple path between any two points.
    Ex1: Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]], Output: 20
    Explanation: We can connect the points as shown above to get the minimum cost of 20.
        Notice that there is a unique path between every pair of points.
    Ex2: Input: points = [[3,12],[-2,5],[-4,1]], Output: 18
    Minimum Spanning Tree - Prim's -> there also kruskals
    """
    """
    Hints: build a map and check min
    """
    def minCostConnectPoints(self, points: list[list[int]]) -> int:
        dist, res = {(x, y): float('inf') if i else 0 for i, (x, y) in enumerate(points)}, 0
        while dist:
            x, y = min(dist, key=dist.get) # obtain the current min edge
            res += dist.pop((x, y))           # remove the point
            for x1, y1 in dist: dist[(x1, y1)] = min(dist[x1, y1], abs(x-x1)+abs(y-y1))
        return res

    """
    743. Network Delay Time
    You are given a network of n nodes, labeled from 1 to n. You are also given times, 
    a list of travel times as directed edges times[i] = (ui, vi, wi), 
    where ui is the source node, vi is the target node, 
    and wi is the time it takes for a signal to travel from source to target.
    We will send a signal from a given node k. 
    Return the minimum time it takes for all the n nodes to receive the signal. 
    If it is impossible for all the n nodes to receive the signal, return -1.
    Ex1: Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2, Output: 2
    Ex2: Input: times = [[1,2,1]], n = 2, k = 1, Output: 1
    Ex3: Input: times = [[1,2,1]], n = 2, k = 2, Output: -1
    O(E * logV) - Dijskra's Algo
    """
    """
    Hints: BFS, build a map, minH
    """
    def networkDelayTime(self, times: list[list[int]], n: int, k: int) -> int:
        return 0

    """
    787. Cheapest Flights Within K Stops
    There are n cities connected by some number of flights. 
    You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that 
    there is a flight from city fromi to city toi with cost pricei.
    You are also given three integers src, dst, and k, 
    return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.
    Ex1: Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 
        src = 0, dst = 3, k = 1, Output: 700
    Explanation:
        The graph is shown above.
        The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
        Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.
    Bellman-Ford - O(E * K) 
    """
    """
    Hints
    """
    def findCheapestPrice(self, n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
        return 0

    """
    1514. Path with Maximum Probability
    You are given an undirected weighted graph of n nodes (0-indexed), 
    represented by an edge list where edges[i] = [a, b] is an undirected edge connecting 
    the nodes a and b with a probability of success of traversing that edge succProb[i].
    Given two nodes start and end, find the path with the maximum probability 
    of success to go from start to end and return its success probability.
    If there is no path from start to end, return 0. 
    Your answer will be accepted if it differs from the correct answer by at most 1e-5.
    """
    """
    Hints: BFS, build adjList, pq, visit
    """
    def maxProbability(self, n: int, edges: list[list[int]], succProb: list[float],
                       start_node: int, end_node: int) -> float:
        return 0

    """
    332. Reconstrust Itinerary  - HARD
    You are given a list of airline tickets where tickets[i] = [fromi, toi] represent 
    the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.
    All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". 
    If there are multiple valid itineraries, you should return the itinerary 
    that has the smallest lexical order when read as a single string.
    For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
    You may assume all tickets form at least one valid itinerary. 
    You must use all the tickets once and only once.
    Ex1: Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
        Output: ["JFK","MUC","LHR","SFO","SJC"]
    Ex2: Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
        Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
        Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] 
        but it is larger in lexical order.
    """
    """
    Hints: BFS, with a map, stack and res
    """
    def findItinetrary(self, tickets: list[list[str]]) -> list[str]:
        return None

    """
    778. Swim in Rising Water - HARD
    You are given an n x n integer matrix grid where each value grid[i][j] 
    represents the elevation at that point (i, j).
    The rain starts to fall. At time t, the depth of the water everywhere is t. 
    You can swim from a square to another 4-directionally adjacent square if and only if 
    the elevation of both squares individually are at most t. 
    You can swim infinite distances in zero time. 
    Of course, you must stay within the boundaries of the grid during your swim.
    Return the least time until you can reach the bottom right square (n - 1, n - 1) 
    if you start at the top left square (0, 0).
    Ex1: Input: grid = [[0,2],[1,3]], Output: 3
    Explanation:
        At time 0, you are in grid location (0, 0).
        You cannot go anywhere else because 4-directionally 
        adjacent neighbors have a higher elevation than t = 0.
        You cannot reach point (1, 1) until time 3.
        When the depth of water is 3, we can swim anywhere inside the grid.
    Ex2: Input: grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
        Output: 16
    Explanation: The final route is shown.
        We need to wait until time 16 so that (0, 0) and (4, 4) are connected.
        O(N^2 logN) - Dijskars
    """
    """
    Hints: BFS, pq = [max(time/height), r, c]
    """
    def swimInWater(self, grid: list[list[int]]) -> int:
        return 0

    """
    892. Alien Dictionary - HARD
    There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. 
    You receive a list of non-empty words from the dictionary, where words are sorted lexicographically 
    by the rules of this new language. Derive the order of letters in this language.
    You may assume all letters are in lowercase.
    The dictionary is invalid, if string a is prefix of string b and b is appear before a.
    If the order is invalid, return an empty string.
    There may be multiple valid order of letters, return the smallest in normal lexicographical order.
    The letters in one string are of the same rank by default and are sorted in Human dictionary order.
    Ex1: Input：["wrt","wrf","er","ett","rftt"], Output："wertf"
        Explanation：
        from "wrt"and"wrf" ,we can get 't'<'f'
        from "wrt"and"er" ,we can get 'w'<'e'
        from "er"and"ett" ,we can get 'r'<'t'
        from "ett"and"rftt" ,we can get 'e'<'r'
        So return "wertf"
    Ex2: Input：["z","x"], Output："zx"
    Explanation：from "z" and "x"，we can get 'z' < 'x', So return "zx"
    Topological Sort
    """
    """
    Hints
    """
    def alienOrder(self, words: list[str]) -> str:
        return ""

    """
    2421. Number of Good Paths
    There is a tree (i.e. a connected, undirected graph with no cycles) 
    consisting of n nodes numbered from 0 to n - 1 and exactly n - 1 edges.
    You are given a 0-indexed integer array vals of length n where vals[i] denotes the value of the ith node. 
    You are also given a 2D integer array edges where edges[i] = [ai, bi] denotes 
    that there exists an undirected edge connecting nodes ai and bi.
    A good path is a simple path that satisfies the following conditions:
    The starting node and the ending node have the same value.
    All nodes between the starting node and the ending node have values less than or equal to 
    the starting node (i.e. the starting node's value should be the maximum value along the path).
    Return the number of distinct good paths.
    Note that a path and its reverse are counted as the same path. For example, 0 -> 1 is considered to be 
    the same as 1 -> 0. A single node is also considered as a valid path.
    Ex1: Input: vals = [1,3,2,1,3], edges = [[0,1],[0,2],[2,3],[2,4]], Output: 6
    Explanation: There are 5 good paths consisting of a single node.
    There is 1 additional good path: 1 -> 0 -> 2 -> 4.
    (The reverse path 4 -> 2 -> 0 -> 1 is treated as the same as 1 -> 0 -> 2 -> 4.)
    Note that 0 -> 2 -> 3 is not a good path because vals[2] > vals[0].
    Ex2: Input: vals = [1,1,2,2,3], edges = [[0,1],[1,2],[2,3],[2,4]], Output: 7
    Explanation: There are 5 good paths consisting of a single node.
    There are 2 additional good paths: 0 -> 1 and 2 -> 3.
    Ex3: Input: vals = [1], edges = [], Output: 1
    Explanation: The tree consists of only one node, so there is one good path.
    """
    """
    Hints: 
    """
    def numberOfGoodPaths(self, vals: list[int], edges: list[list[int]]) -> int:
        return 0

    """
    1579. Remove Max Number of Edges to Keep Graph Fully Traversable
    Alice and Bob have an undirected graph of n nodes and three types of edges:
        Type 1: Can be traversed by Alice only.
        Type 2: Can be traversed by Bob only.
        Type 3: Can be traversed by both Alice and Bob.
    Given an array edges where edges[i] = [typei, ui, vi] represents a bidirectional edge of type typei between nodes
    ui and vi, find the maximum number of edges you can remove so that after removing the edges, 
    the graph can still be fully traversed by both Alice and Bob. 
    The graph is fully traversed by Alice and Bob if starting from any node, they can reach all other nodes.
    Return the maximum number of edges you can remove, or return -1 if Alice and Bob cannot fully traverse the graph.
    Ex1: Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]], Output: 2
    Explanation: If we remove the 2 edges [1,1,2] and [1,1,3]. The graph will still be fully traversable by Alice and Bob. 
    Removing any additional edge will not make it so. So the maximum number of edges we can remove is 2.
    Ex2: Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,4],[2,1,4]], Output: 0
    Explanation: Notice that removing any edge will not make the graph fully traversable by Alice and Bob.
    Ex3: Input: n = 4, edges = [[3,2,3],[1,1,2],[2,3,4]], Output: -1
    Explanation: In the current graph, Alice cannot reach node 4 from the other nodes. 
    Likewise, Bob cannot reach 1. Therefore it's impossible to make the graph fully traversable.
    """
    def maxNumEdgesToRemove(self, n: int, edges: list[list[int]]) -> int:
        return 0

    """
    1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree - (HARD)
    Given a weighted undirected connected graph with n vertices numbered from 0 to n - 1, and an array edges 
    where edges[i] = [ai, bi, weighti] represents a bidirectional and weighted edge between nodes ai and bi. 
    A minimum spanning tree (MST) is a subset of the graph's edges that connects all vertices without cycles 
    and with the minimum possible total edge weight.
    Find all the critical and pseudo-critical edges in the given graph's minimum spanning tree (MST). 
    An MST edge whose deletion from the graph would cause the MST weight to increase is called a critical edge. 
    On the other hand, a pseudo-critical edge is that which can appear in some MSTs but not all.
    Note that you can return the indices of the edges in any order.
    Ex1: Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
    Output: [[0,1],[2,3,4,5]]
    Ex2: Input: n = 4, edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]], Output: [[],[0,1,2,3]]
    """
    """
    Hints: MST, UF
    """
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: list[list[int]]) -> list[list[int]]:
        return None

    #################### 1-D Dynamic Programming ############# - 33
    """
    - Dymanic Programming: Top-down memoization
    def fn(arr):
        def dp(STATE):
            if BASE_CASE: return 0
            if STATE in memo: return memo[STATE]
            ans = RECURRENCE_RELATION(STATE)
            memo[STATE] = ans
            return ans
        memo = {}
    return dp(STATE_FOR_WHOLE_INPUT)
    """

    """
    70. Climbing Stairs
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. 
    In how many distinct ways can you climb to the top?
    Ex1: Input: n = 2, Output: 2
    Explanation: There are two ways to climb to the top.
        1. 1 step + 1 step
        2. 2 steps
    Ex2: Input: n = 3, Output: 3
    Explanation: There are three ways to climb to the top.
        1. 1 step + 1 step + 1 step
        2. 1 step + 2 steps
        3. 2 steps + 1 step
    """
    """
    Hints: from 2 - n+1, use temp to save curr
    """
    def climbStairs(self, n: int) -> int:
        if n == 0: return 0
        res = 0
        def dfs(n):
            if n == 1: return 1
            if n == 2: return 2
            return dfs()


    """
    746. Min Cost Climbing Stairs
    You are given an integer array cost where cost[i] is the cost of ith step on a staircase. 
    Once you pay the cost, you can either climb one or two steps.
    You can either start from the step with index 0, or the step with index 1.
    Return the minimum cost to reach the top of the floor.
    Ex1: Input: cost = [10,15,20], Output: 15
    Explanation: You will start at index 1.
        - Pay 15 and climb two steps to reach the top.
        The total cost is 15.
    Ex2: Input: cost = [1,100,1,1,1,100,1,1,100,1], Output: 6
    Explanation: You will start at index 0.
        - Pay 1 and climb two steps to reach index 2.
        - Pay 1 and climb two steps to reach index 4.
        - Pay 1 and climb two steps to reach index 6.
        - Pay 1 and climb one step to reach index 7.
        - Pay 1 and climb two steps to reach index 9.
        - Pay 1 and climb one step to reach the top.
        The total cost is 6.
    """
    """
    Hints: build dp, check the cost and the min at the dp, then return check/return min 
    """
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        return 0

    """
    198. House Robber
    You are a professional robber planning to rob houses along a street. 
    Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses 
    have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given an integer array nums representing the amount of money of each house, 
    return the maximum amount of money you can rob tonight without alerting the police.
    Ex1: Input: nums = [1,2,3,1], Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.
    Ex2: Input: nums = [2,7,9,3,1], Output: 12
    Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
    Total amount you can rob = 2 + 9 + 1 = 12.
    """
    """
    Hints: hold the first elem, then get the max
    """
    def rob(self, nums: list[int]) -> int:
        return 0

    """
    213. House Robber II
    You are a professional robber planning to rob houses along a street. 
    Each house has a certain amount of money stashed. 
    All houses at this place are arranged in a circle. 
    That means the first house is the neighbor of the last one. Meanwhile, 
    adjacent houses have a security system connected, 
    and it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given an integer array nums representing the amount of money of each house, 
    return the maximum amount of money you can rob tonight without alerting the police.
    Ex1: Input: nums = [2,3,2], Output: 3
    Explanation: You cannot rob house 1 (money = 2) 
    and then rob house 3 (money = 2), because they are adjacent houses.
    Ex2: Input: nums = [1,2,3,1], Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.
    """
    """
    Hints: build helper function then pass in the max
    """
    def robII(self, nums: list[int]) -> int:
        return 0

    """
    5. Lonest Palindromic Substring
    Given a string s, return the longest palindromic substring in s
    Ex1: Input: s = "babad", Output: "bab" - Explanation: "aba" is also a valid answer.
    Ex2: Input: s = "cbbd", Output: "bb"
    """
    """
    Hints: i, i for even, i, i+1 for odd
    """
    def longestPalindrome(self, s: str) -> str:
        return ""

    """
    647. Palindromic Substrings
    Given a string s, return the number of palindromic substrings in it.
    A string is a palindrome when it reads the same backward as forward.
    A substring is a contiguous sequence of characters within the string.
    Ex1: Input: s = "abc", Output: 3
    Explanation: Three palindromic strings: "a", "b", "c".
    Ex2: Input: s = "aaa", Output: 6
    Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
    """
    """
    Hints: dfs, for i, i+1
    """
    def countSubstrings(self, s: str) -> int:
        return 0

    """
    300. Longest Increasing Subsequence
    Given an integer array nums, return the length of the longest strictly increasing subsequence
    Ex1: Input: nums = [10,9,2,5,3,7,101,18], Output: 4
    Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
    Ex2: Input: nums = [0,1,0,3,2,3], Output: 4
    Ex3: Input: nums = [7,7,7,7,7,7,7], Output: 1
    """
    """
    Hints: Binary Search, l=0, r=res
    """
    def lengthOfLIS(self, nums: list[int]) -> int:
        dp, res = [0]*len(nums), 0
        for num in nums:
            l, r = 0, res
            while l != r:
                mid = l + (r-l) // 2
                if dp[mid] < num: l = mid+1
                else: r = mid
            res = max(res, l+1)
            dp[l] = num
        return res

    # new Picks from 300 list

    """
    120. Triangle
    Given a triangle array, return the minimum path sum from top to bottom.
    For each step, you may move to an adjacent number of the row below. 
    More formally, if you are on index i on the current row, 
    you may move to either index i or index i + 1 on the next row.
    Ex1: Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]], Output: 11
    Explanation: The triangle looks like:
       2
      3 4
     6 5 7
    4 1 8 3 , The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
    Ex2: Input: triangle = [[-10]], Output: -10
    """
    """
    Hints: bottom up, take the last row as the dp, then update dp
    """
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        dp = triangle[-1]
        for r in range(len(triangle)-2, -1, -1):
            for c in range(0, r+1):
                dp[c] = triangle[r][c] + min(dp[c], dp[c+1])
        return dp[0]

    """
    740. Delete and Earn
    You are given an integer array nums. You want to maximize the number of points you 
    get by performing the following operation any number of times:
    Pick any nums[i] and delete it to earn nums[i] points. 
    Afterwards, you must delete every element equal to nums[i] - 1 and every element equal to nums[i] + 1.
    Return the maximum number of points you can earn by applying the above operation some number of times.
    Ex1: Input: nums = [3,4,2], Output: 6
    Explanation: You can perform the following operations:
    - Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
    - Delete 2 to earn 2 points. nums = []. You earn a total of 6 points.
    Ex2: Input: nums = [2,2,3,3,3,4], Output: 9
    Explanation: You can perform the following operations:
    - Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums = [3,3].
    - Delete a 3 again to earn 3 points. nums = [3].
    - Delete a 3 once more to earn 3 points. nums = []. You earn a total of 9 points.
    """
    """
    Hints: need a store and dp with len of max(nums)+1, add num to store
    """
    def deleteAndEarn(self, nums: list[int]) -> int:
        return 0

    """
    377. Combination Sum IV
    Given an array of distinct integers nums and a target integer target, 
    return the number of possible combinations that add up to target.
    The test cases are generated so that the answer can fit in a 32-bit integer.
    Ex1: Input: nums = [1,2,3], target = 4, Output: 7
    Explanation: The possible combination ways are:
    (1, 1, 1, 1)
    (1, 1, 2)
    (1, 2, 1)
    (1, 3)
    (2, 1, 1)
    (2, 2)
    (3, 1)
    Note that different sequences are counted as different combinations.
    Ex2: Input: nums = [9], target = 3, Output: 0
    """
    """
    Hints: 
    """
    def combinationSum4(self, nums: list[int], target: int) -> int:
        return 0

    #################### 2-D Dynamic Programming ############# /11 (4 HARD) -

    """
    62. unique Paths
    There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
    The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
    The robot can only move either down or right at any point in time.
    Given the two integers m and n, return the number of possible unique paths that the robot can take 
    to reach the bottom-right corner.
    The test cases are generated so that the answer will be less than or equal to 2 * 10^9.
    Ex1: Input: m = 3, n = 7, Output: 28
    Ex2: Input: m = 3, n = 2, Output: 3
    Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
        1. Right -> Down -> Down
        2. Down -> Down -> Right
        3. Down -> Right -> Down
    """
    """
    Hints: build a row, and a newRow bottom-up
    """
    def uniquePath(self, m: int, n: int) -> int:
        return 0

    """
    1143. Longest Common Subsequence
    Given two strings text1 and text2, return the length of their longest common subsequence. 
    If there is no common subsequence, return 0.
    A subsequence of a string is a new string generated from the original string with 
    some characters (can be none) deleted without changing the relative order of the remaining characters.
    For example, "ace" is a subsequence of "abcde".
    A common subsequence of two strings is a subsequence that is common to both strings.
    Ex1: Input: text1 = "abcde", text2 = "ace" , Output: 3  
        Explanation: The longest common subsequence is "ace" and its length is 3.
    Ex2: Input: text1 = "abc", text2 = "abc", Output: 3
        Explanation: The longest common subsequence is "abc" and its length is 3.
    Ex3: Input: text1 = "abc", text2 = "def", Output: 0
        Explanation: There is no such common subsequence, so the result is 0.
    """
    """
    Hints: two dp, iterate bottom up
    """
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        return 0

    """
    309. Best time to Buy and Sell Stocks with CoolDown
    You are given an array prices where prices[i] is the price of a given stock on the ith day.
    Find the maximum profit you can achieve. You may complete as many transactions as you like 
    (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
    After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
    Note: You may not engage in multiple transactions simultaneously 
    (i.e., you must sell the stock before you buy again).
    Ex1: Input: prices = [1,2,3,0,2], Output: 3
        Explanation: transactions = [buy, sell, cooldown, buy, sell]
    Ex2: Input: prices = [1], Output: 0
    """
    """
    Hints: set buy to the max cd - price, cd = sel, sel max buy+price
    """
    def maxProfitII(self, prices: list[int]) -> int:
        return 0

    """
    518. Coin Change II
    You are given an integer array coins representing coins of different denominations 
    and an integer amount representing a total amount of money.
    Return the number of combinations that make up that amount. 
    If that amount of money cannot be made up by any combination of the coins, return 0.
    You may assume that you have an infinite number of each kind of coin.
    The answer is guaranteed to fit into a signed 32-bit integer.
    Ex1: Input: amount = 5, coins = [1,2,5], Output: 4
    Explanation: there are four ways to make up the amount:
        5=5
        5=2+2+1
        5=2+1+1+1
        5=1+1+1+1+1
    Ex2: Input: amount = 3, coins = [2], Output: 0
    Explanation: the amount of 3 cannot be made up just with coins of 2.
    Ex3: Input: amount = 10, coins = [10], Output: 1
    Knapsack problem
    """
    """
    Hints:
    """
    def change(self, amount: int, coins: list[int]) -> int:
        return 0

    """
    494. Target Sum
    You are given an integer array nums and an integer target.
    You want to build an expression out of nums by adding one of 
    the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
    For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and 
    concatenate them to build the expression "+2-1".
    Return the number of different expressions that you can build, which evaluates to target.
    Ex1: Input: nums = [1,1,1,1,1], target = 3, Output: 5
    Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
        -1 + 1 + 1 + 1 + 1 = 3
        +1 - 1 + 1 + 1 + 1 = 3
        +1 + 1 - 1 + 1 + 1 = 3
        +1 + 1 + 1 - 1 + 1 = 3
        +1 + 1 + 1 + 1 - 1 = 3
    Ex2: Input: nums = [1], target = 1, Output: 1
    """
    """
    Hints:
    """
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        if total < abs(target) or (total + target) & 1: return 0
        def knapsack(target):
            dp = [1]+[0]*total
            for num in nums:
                for j in range(total, num-1, -1): dp[j] += dp[j-num]
            return dp[target]
        return knapsack((total+target)//2)

    # new Picks from 300 list

    """
    63. Unique Paths II
    You are given an m x n integer array grid. There is a robot initially located at the top-left corner 
    (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
    The robot can only move either down or right at any point in time.
    An obstacle and space are marked as 1 or 0 respectively in grid. 
    A path that the robot takes cannot include any square that is an obstacle.
    Return the number of possible unique paths that the robot can take to reach the bottom-right corner.
    The testcases are generated so that the answer will be less than or equal to 2 * 10^9.
    Ex1: Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]], Output: 2
    Explanation: There is one obstacle in the middle of the 3x3 grid above.
        There are two ways to reach the bottom-right corner:
        1. Right -> Right -> Down -> Down
        2. Down -> Down -> Right -> Right
    Ex2: Input: obstacleGrid = [[0,1],[0,0]], Output: 1
    """
    """
    Hints: top down, 1D dp, 
    """
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        return 0

    """
    516. Longest Palindromic Subsequence
    Given a string s, find the longest palindromic subsequence's length in s.
    A subsequence is a sequence that can be derived from another sequence by deleting some or 
    no elements without changing the order of the remaining elements.
    Ex1: Input: s = "bbbab", Output: 4
    Explanation: One possible longest palindromic subsequence is "bbbb".
    Ex2: Input: s = "cbbd", Output: 2
    Explanation: One possible longest palindromic subsequence is "bb".
    """
    """
    Hints:build helper, send s and s-reversed, then compare 
    """
    def longestPalindromeSubseq(self, s: str) -> int:
        return 0

    """
    1049. Last Stone Weight II
    You are given an array of integers stones where stones[i] is the weight of the ith stone.
    We are playing a game with the stones. On each turn, we choose any two stones and smash them together. 
    Suppose the stones have weights x and y with x <= y. The result of this smash is:
        If x == y, both stones are destroyed, and
        If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
    At the end of the game, there is at most one stone left.
    Return the smallest possible weight of the left stone. If there are no stones left, return 0.
    Ex1: Input: stones = [2,7,4,1,8,1], Output: 1
    Explanation:
        We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
        we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
        we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
        we can combine 1 and 1 to get 0, so the array converts to [1], then that's the optimal value.
    Ex2: Input: stones = [31,26,33,21,40], Output: 5
    """
    """
    Hints: One-line code functools.reduce 
    """
    def lastStoneWeightII(self, stones: list[int]) -> int:
        return 0

    """
    64. Minimum Path Sum
    Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, 
    which minimizes the sum of all numbers along its path.
    Note: You can only move either down or right at any point in time.
    Ex1: Input: grid = [[1,3,1],[1,5,1],[4,2,1]], Output: 7
    Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
    Ex2: Input: grid = [[1,2,3],[4,5,6]], Output: 12
    """
    """
    Hints: 
    """
    def minPathSum(self, grid: list[list[int]]) -> int:
        M, N, prev = len(grid), len(grid[0]), [float('inf')] * len(grid[0])
        prev[-1] = 0
        for row in range(M-1, -1, -1):
            dp = [float('inf')] * N
            for col in range(N-1, -1, -1):
                if col < N-1: dp[col] = min(dp[col], dp[col+1])
                dp[col] = min(dp[col], prev[col]) + grid[row][col]
            prev = dp
        return prev[0]

    """
    221. Maximal Square
    Given an m x n binary matrix filled with 0's and 1's, 
    find the largest square containing only 1's and return its area.
    Ex1: Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],
                          ["1","1","1","1","1"],["1","0","0","1","0"]], Output: 4
    Ex2: Input: matrix = [["0","1"],["1","0"]], Output: 1
    Ex3: Input: matrix = [["0"]], Output: 0
    """
    """
    Hints:
    """
    def maximalSquare(self, matrix: list[list[str]]) -> int:
        return 0

    #################### Greedy ############################## - 16

    """
    53. Maximum Subarray
    Given an integer array nums, find the contiguous subarray with the largest sum, and return its sum.
    Ex1: Input: nums = [-2,1,-3,4,-1,2,1,-5,4], Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.
    Ex2: Input: nums = [1], Output: 1
    Explanation: The subarray [1] has the largest sum 1.
    Ex3: Input: nums = [5,4,-1,7,8], Output: 23
    Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
    """
    """
    Hints: build res, total, iterate
    """
    def maxSubArray(self, nums: list[int]) -> int:
        return 0

    """
    55. Jump Game
    You are given an integer array nums. You are initially positioned at the array's first index, 
    and each element in the array represents your maximum jump length at that position.
    Return true if you can reach the last index, or false otherwise.
    Ex1: Input: nums = [2,3,1,1,4], Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    Ex2: Input: nums = [3,2,1,0,4], Output: false
    Explanation: You will always arrive at index 3 no matter what. 
    Its maximum jump length is 0, which makes it impossible to reach the last index.
    """
    """
    Hints: iterate backwards, check the goal
    """
    def canJump(self, nums: list[int]) -> bool:
        return False

    """
    45. Jump Game II
    You are given a 0-indexed array of integers nums of length n. 
    You are initially positioned at nums[0].
    Each element nums[i] represents the maximum length of a forward jump from index i. 
    In other words, if you are at nums[i], you can jump to any nums[i + j] where:
    0 <= j <= nums[i] and
    i + j < n
    Return the minimum number of jumps to reach nums[n - 1]. 
    The test cases are generated such that you can reach nums[n - 1].
    Ex1: Input: nums = [2,3,1,1,4], Output: 2
    Explanation: The minimum number of jumps to reach the last index is 2. 
        Jump 1 step from index 0 to 1, then 3 steps to the last index.
    Ex2: Input: nums = [2,3,0,1,4], Output: 2
    """
    """
    Hints:
    """
    def jump(self, nums: list[int]) -> int:
        return 0

    """
    134. Gas Station
    There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
    You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to 
    its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.
    Given two integer arrays gas and cost, return the starting gas station's index if you can travel around 
    the circuit once in the clockwise direction, otherwise return -1. 
    If there exists a solution, it is guaranteed to be unique
    Ex1: Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2], Output: 3
    Explanation:
        Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
        Travel to station 4. Your tank = 4 - 1 + 5 = 8
        Travel to station 0. Your tank = 8 - 2 + 1 = 7
        Travel to station 1. Your tank = 7 - 3 + 2 = 6
        Travel to station 2. Your tank = 6 - 4 + 3 = 5
        Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
        Therefore, return 3 as the starting index.
    Ex2: Input: gas = [2,3,4], cost = [3,4,3], Output: -1
    Explanation:
        You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
        Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
        Travel to station 0. Your tank = 4 - 3 + 2 = 3
        Travel to station 1. Your tank = 3 - 3 + 3 = 3
        You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
        Therefore, you can't travel around the circuit once no matter where you start.
    """
    """
    Hints: single pass, gas - cost, 
    """
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        return 0

    """
    846. Hand of Straights
    Alice has some number of cards and she wants to rearrange the cards into groups 
    so that each group is of size groupSize, and consists of groupSize consecutive cards.
    Given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize, 
    return true if she can rearrange the cards, or false otherwise.
    Ex1: Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3, Output: true
    Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
    Ex2: Input: hand = [1,2,3,4,5], groupSize = 4, Output: false
    Explanation: Alice's hand can not be rearranged into groups of 4.
    """
    """
    Hints: map, q, BFS
    """
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

    """
    1899. Merge Triplets to Form Target Triplet
    A triplet is an array of three integers. You are given a 2D integer array triplets, 
    where triplets[i] = [ai, bi, ci] describes the ith triplet. 
    You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.
    To obtain target, you may apply the following operation on triplets any number of times (possibly zero):
    Choose two indices (0-indexed) i and j (i != j) and update triplets[j] 
    to become [max(ai, aj), max(bi, bj), max(ci, cj)].
    For example, if triplets[i] = [2, 5, 3] and triplets[j] = [1, 7, 5], triplets[j] 
    will be updated to [max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5].
    Return true if it is possible to obtain the target triplet [x, y, z] as an 
    element of triplets, or false otherwise.
    Ex1: Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5], Output: true
    Explanation: Perform the following operations:
        - Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. 
        Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. 
        triplets = [[2,5,3],[1,8,4],[2,7,5]]
        The target triplet [2,7,5] is now an element of triplets.
    Ex2: Input: triplets = [[3,4,5],[4,5,6]], target = [3,2,5], Output: false
    Explanation: It is impossible to have [3,2,5] as an element because there is no 2 in any of the triplets.
    Ex3: Input: triplets = [[2,5,3],[2,3,4],[1,2,5],[5,2,3]], target = [5,5,5], Output: true
    Explanation: Perform the following operations:
        - Choose the first and third triplets [[2,5,3],[2,3,4],[1,2,5],[5,2,3]]. 
        Update the third triplet to be [max(2,1), max(5,2), max(3,5)] = [2,5,5]. 
        triplets = [[2,5,3],[2,3,4],[2,5,5],[5,2,3]].
        - Choose the third and fourth triplets [[2,5,3],[2,3,4],[2,5,5],[5,2,3]]. 
        Update the fourth triplet to be [max(2,5), max(5,2), max(5,3)] = [5,5,5]. 
        triplets = [[2,5,3],[2,3,4],[2,5,5],[5,5,5]].
        The target triplet [5,5,5] is now an element of triplets.
    """
    """
    Hints: build a set, check indices
    """
    def mergeTriplets(self, triplets: list[list[int]], target: list[int]) -> bool:
        return False

    """
    763. Partition Labels
    You are given a string s. We want to partition the string into as many parts as possible 
    so that each letter appears in at most one part.
    Note that the partition is done so that after concatenating all the parts in order, 
    the resultant string should be s.
    Return a list of integers representing the size of these parts.
    Ex1: Input: s = "ababcbacadefegdehijhklij", Output: [9,7,8]
    Explanation:
           The partition is "ababcbaca", "defegde", "hijhklij".
           This is a partition so that each letter appears in at most one part.
           A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
    Ex2: Input: s = "eccbbbbdec", Output: [10]
    """
    """
    Hints: build map of lastInd, then iterate end=max, 
    """
    def partitionLabels(self, s: str) -> list[int]:
        return None

    """
    678. Valid Parenthesis String
    Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
    The following rules define a valid string:
    Any left parenthesis '(' must have a corresponding right parenthesis ')'.
    Any right parenthesis ')' must have a corresponding left parenthesis '('.
    Left parenthesis '(' must go before the corresponding right parenthesis ')'.
    '*' could be treated as a single right parenthesis ')' or a 
    single left parenthesis '(' or an empty string "".
    Ex1: Input: s = "()", Output: true
    Ex2: Input: s = "(*)", Output: true
    Ex3: Input: s = "(*))", Output: true
    """
    """
    Hints:
    """
    def checkVaildString(self, s: str) -> bool:
        return False

    # new Picks from 300 list

    """
    918. Maximum Sum Circular Subarray
    Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
    A circular array means the end of the array connects to the beginning of the array. 
    Formally, the next element of nums[i] is nums[(i + 1) % n] 
    and the previous element of nums[i] is nums[(i - 1 + n) % n].
    A subarray may only include each element of the fixed buffer nums at most once. 
    Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], 
    there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.
    Ex1: Input: nums = [1,-2,3,-2], Output: 3, Explanation: Subarray [3] has maximum sum 3.
    Ex2: Input: nums = [5,-3,5], Output: 10, Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.
    Ex3: Input: nums = [-3,-2,-3], Output: -2, Explanation: Subarray [-2] has maximum sum -2.
    """
    def maxSubarraySumCircular(self, nums: list[int]) -> int:
        totalSum = 0
        currMaxSum = 0
        currMinSum = 0
        maxSum = -math.inf
        minSum = math.inf

        for num in nums:
            totalSum += num
            currMaxSum = max(currMaxSum + num, num)
            currMinSum = min(currMinSum + num, num)
            maxSum = max(maxSum, currMaxSum)
            minSum = min(minSum, currMinSum)

        return maxSum if maxSum < 0 else max(maxSum, totalSum - minSum)

    """
    978. Longest Turbulent Subarray
    Given an integer array arr, return the length of a maximum size turbulent subarray of arr.
    A subarray is turbulent if the comparison sign flips between each adjacent pair of elements in the subarray.
    More formally, a subarray [arr[i], arr[i + 1], ..., arr[j]] of arr is said to be turbulent if and only if:
    For i <= k < j:
    arr[k] > arr[k + 1] when k is odd, and
    arr[k] < arr[k + 1] when k is even.
    Or, for i <= k < j:
    arr[k] > arr[k + 1] when k is even, and
    arr[k] < arr[k + 1] when k is odd.
    Ex1: Input: arr = [9,4,2,10,7,8,8,1,9], Output: 5, Explanation: arr[1] > arr[2] < arr[3] > arr[4] < arr[5]
    Ex2: Input: arr = [4,8,12,16], Output: 2
    Ex3: Input: arr = [100], Output: 1
    """
    def maxTurbulenceSize(self, arr: list[int]) -> int:
        return 0

    #################### Intervals ########################### - 8

    """
    920. Meeting Rooms
    Given an array of meeting time intervals consisting of start and end times 
    [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.
    Ex1: Input: intervals = [(0,30),(5,10),(15,20)], Output: false
        Explanation: (0,30), (5,10) and (0,30),(15,20) will conflict
    Ex2: Input: intervals = [(5,8),(9,15)], Output: true
        Explanation: Two times will not conflict 
    """
    """
    Hints: sort then check
    """
    def canAttend(self, intervals: list[list[int]]) -> bool:
        return False

    """
    919. Meeting Rooms II
    Given an array of meeting time intervals consisting of start and end times 
    [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.)
    Ex1: Input: intervals = [(0,30),(5,10),(15,20)], Output: 2
        Explanation: We need two meeting rooms, room1: (0,30), room2: (5,10),(15,20)
    Ex2: Input: intervals = [(2,7)], Output: 1
        Explanation: Only need one meeting room
    """
    """
    Hints: build time array with start, end, then get the count and maxcount
    """
    def minMeetingRooms(self, intervals: list[list[int]]) -> int:
        return 0

    """
    56. Merge Intervals
    Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
    and return an array of the non-overlapping intervals that cover all the intervals in the input.
    Ex1: Input: intervals = [[1,3],[2,6],[8,10],[15,18]], Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
    Ex2: Input: intervals = [[1,4],[4,5]], Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.
    """
    """
    Hints: sort, build res array, then compare
    """
    def mergeIntervals(self, intervals: list[list[int]]) -> list[list[int]]:
        return None

    """
    57. Insert Interval
    You are given an array of non-overlapping intervals where 
    intervals[i] = [starti, endi] represent the start and the end of the ith interval 
    and intervals is sorted in ascending order by starti. 
    You are also given an interval newInterval = [start, end] 
    that represents the start and end of another interval.
    Insert newInterval into intervals such that intervals is still sorted in ascending order 
    by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
    Return intervals after the insertion.
    Ex1: Input: intervals = [[1,3],[6,9]], newInterval = [2,5], Output: [[1,5],[6,9]]
    Ex2: Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8], 
    Output: [[1,2],[3,10],[12,16]]
    Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
    """
    """
    Hints: Use the newInterval as anker to append and merge
    """
    def insertInterval(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
        return None

    """
    435. None-overlapping Intervals
    Given an array of intervals intervals where intervals[i] = [starti, endi], 
    return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
    Ex1: Input: intervals = [[1,2],[2,3],[3,4],[1,3]], Output: 1
    Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
    Ex2: Input: intervals = [[1,2],[1,2],[1,2]], Output: 2
    Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
    Ex3: Input: intervals = [[1,2],[2,3]], Output: 0
    Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
    """
    """
    Hints: sort, need prev, count, check start and end
    """
    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        return 0

    """
    1851. Minimum Interval To include Each Query - HARD
    You are given a 2D integer array intervals, where intervals[i] = [lefti, righti] 
    describes the ith interval starting at lefti and ending at righti (inclusive). 
    The size of an interval is defined as the number of integers it contains, 
    or more formally righti - lefti + 1.
    You are also given an integer array queries. 
    The answer to the jth query is the size of the smallest interval i such that 
    lefti <= queries[j] <= righti. If no such interval exists, the answer is -1.
    Return an array containing the answers to the queries.
    Ex1: Input: intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5], Output: [3,3,1,4]
    Explanation: The queries are processed as follows:
        - Query = 2: The interval [2,4] is the smallest interval containing 2. The answer is 4 - 2 + 1 = 3.
        - Query = 3: The interval [2,4] is the smallest interval containing 3. The answer is 4 - 2 + 1 = 3.
        - Query = 4: The interval [4,4] is the smallest interval containing 4. The answer is 4 - 4 + 1 = 1.
        - Query = 5: The interval [3,6] is the smallest interval containing 5. The answer is 6 - 3 + 1 = 4.
    Ex2: Input: intervals = [[2,3],[2,5],[1,8],[20,25]], queries = [2,19,5,22], Output: [2,-1,4,6]
    Explanation: The queries are processed as follows:
        - Query = 2: The interval [2,3] is the smallest interval containing 2. The answer is 3 - 2 + 1 = 2.
        - Query = 19: None of the intervals contain 19. The answer is -1.
        - Query = 5: The interval [2,5] is the smallest interval containing 5. The answer is 5 - 2 + 1 = 4.
        - Query = 22: The interval [20,25] is the smallest interval containing 22. The answer is 25 - 20 + 1 = 6.
    """
    """
    Hints: sort, minH, hash, i, treat it like BFS
    """
    def minInterval(self, intervals: list[list[int]], queries: list[int]) -> list[int]:
        return None

    # new Picks from 300 list - last 2

    """
    1288. Remove Covered Intervals
    Given an array intervals where intervals[i] = [li, ri] represent the interval [li, ri), 
    remove all intervals that are covered by another interval in the list.
    The interval [a, b) is covered by the interval [c, d) if and only if c <= a and b <= d.
    Return the number of remaining intervals.
    Ex1: Input: intervals = [[1,4],[3,6],[2,8]], Output: 2
        Explanation: Interval [3,6] is covered by [2,8], therefore it is removed.
    Ex2: Input: intervals = [[1,4],[2,3]], Output: 1
    """
    """
    Hints: res to the len, long=0, sort, then check ends
    """
    def removeCoveredIntervals(self, intervals: list[list[int]]) -> int:
        return 0

    """
    352. Data Stream as Disjoint Intervals
    SummaryRange CLASS - HARD
    """

    #################### Math & Geometry ##################### - 22

    """
    202. Happy Number
    Write an algorithm to determine if a number n is happy.
    A happy number is a number defined by the following process:
    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), 
    or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.
    Return true if n is a happy number, and false if not.
    Ex1: Input: n = 19, Output: true
    Explanation:
        12 + 92 = 82
        82 + 22 = 68
        62 + 82 = 100
        12 + 02 + 02 = 1
    Ex2: Input: n = 2, Output: False 
    """
    """
    Hints: Use a set, then sum
    """
    def isHappy(self, n: int) -> bool:
        return False

    """
    66. Plus One
    You are given a large integer represented as an integer array digits, 
    where each digits[i] is the ith digit of the integer. 
    The digits are ordered from most significant to least significant in left-to-right order. 
    The large integer does not contain any leading 0's.
    Increment the large integer by one and return the resulting array of digits.
    Ex1: Input: digits = [1,2,3], Output: [1,2,4]
    Explanation: The array represents the integer 123.
        Incrementing by one gives 123 + 1 = 124.
        Thus, the result should be [1,2,4].
    Ex2: Input: digits = [4,3,2,1], Output: [4,3,2,2]
    Explanation: The array represents the integer 4321.
        Incrementing by one gives 4321 + 1 = 4322.
        Thus, the result should be [4,3,2,2].
    Ex3:  Input: digits = [9], Output: [1,0]
    Explanation: The array represents the integer 9.
        Incrementing by one gives 9 + 1 = 10.
        Thus, the result should be [1,0].
    """
    """
    Hints: conv to str, then add one
    """
    def plusOne(self, digits: list[int]) -> list[int]:
        return None

    """
    48. Rotate Image
    You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
    You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. 
    DO NOT allocate another 2D matrix and do the rotation.
    Ex1: Input: matrix = [[1,2,3],[4,5,6],[7,8,9]], Output: [[7,4,1],[8,5,2],[9,6,3]]
    Ex2: Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
        Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    """
    """
    Hints: 2-for loops, append what you pop
    """
    def rotate(self, matrix: list[list[int]]) -> None:
        return None

    """
    54. Spiral Matrix
    Given an m x n matrix, return all elements of the matrix in spiral order.
    Ex1: Input: matrix = [[1,2,3],[4,5,6],[7,8,9]], Output: [1,2,3,6,9,8,7,4,5]
    Ex2: Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]], Output: [1,2,3,4,8,12,11,10,9,5,6,7]
    """
    """
    Hints: l, r, t, b
    """
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        return None

    """
    73. Set Matrix Zeroes
    Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    You must do it in place.
    Ex1: Input: matrix = [[1,1,1],[1,0,1],[1,1,1]], Output: [[1,0,1],[0,0,0],[1,0,1]]
    Ex2: Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]], Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
    """
    """
    Hints: get a list of indices then set it
    """
    def setZeroes(self, matrix: list[list[int]]) -> None:
        ROWS, COLS = len(matrix), len(matrix[0])
        res = [[i, j] for i in range(ROWS) for j in range(COLS) if matrix[i][j] == 0]
        for k, l in res:
            for row in range(COLS): matrix[k][row] = 0
            for col in range(ROWS): matrix[col][l] = 0

    """
    50. Pow(x, n)
    Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
    Ex1: Input: x = 2.00000, n = 10, Output: 1024.00000
    Ex2: Input: x = 2.10000, n = 3, Output: 9.26100
    Ex3: Input: x = 2.00000, n = -2, Output: 0.25000
        Explanation: 2-2 = 1/22 = 1/4 = 0.25
    """
    """
    Hints:
    """
    def myPow(self, x: float, n: int) -> float:
        return 0.0

    """
    43. Multiply Strings
    Given two non-negative integers num1 and num2 represented as strings, 
    return the product of num1 and num2, also represented as a string.
    Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
    Ex1: Input: num1 = "2", num2 = "3", Output: "6"
    Ex2: Input: num1 = "123", num2 = "456", Output: "56088"
    """
    """
    Hints: build a dict, 
    """
    def multiply(self, num1: str, num2: str) -> str:
        num = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
        r1, r2 = 0, 0
        for i in num1: r1 = 10 * r1 + num[i]
        for i in num2: r2 = 10 * r2 + num[i]
        return str(r1*r2)

    """
    2013. Detect Squares - DetectSquares CLASS
    """

    # new Picks from 300 list

    """
    9. Palindrome Number
    Given an integer x, return true if x is a palindrome, and false otherwise.
    Ex1: Input: x = 121, Output: true
        Explanation: 121 reads as 121 from left to right and from right to left.
    Ex2: Input: x = -121, Output: false
        Explanation: From left to right, it reads -121. From right to left, it becomes 121-. 
        Therefore it is not a palindrome.
    Ex3: Input: x = 10, Output: false
    Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
    """
    def numIsPanlindrome(self, x: int) -> bool:
        return False

    """
    12. Integer To Roman
    Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
    Symbol       Value
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000
    For example, 2 is written as II in Roman numeral, just two one's added together. 
    12 is written as XII, which is simply X + II. 
    The number 27 is written as XXVII, which is XX + V + II.
    Roman numerals are usually written largest to smallest from left to right. 
    However, the numeral for four is not IIII. Instead, the number four is written as IV. 
    Because the one is before the five we subtract it making four. 
    The same principle applies to the number nine, which is written as IX. 
    There are six instances where subtraction is used:
    I can be placed before V (5) and X (10) to make 4 and 9. 
    X can be placed before L (50) and C (100) to make 40 and 90. 
    C can be placed before D (500) and M (1000) to make 400 and 900.
    Given an integer, convert it to a roman numeral.
    Ex1: Input: num = 3, Output: "III"
        Explanation: 3 is represented as 3 ones.
    Ex2: Input: num = 58, Output: "LVIII"
        Explanation: L = 50, V = 5, III = 3.
    Ex3: Input: num = 1994, Output: "MCMXCIV"
        Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
    """
    def intToRoman(self, num: int) -> str:
        return ""

    #################### Bit Manipulation #################### - 10
    """
    Overall Notes: x=10 ,y=4
    AND: x & y =0   -> Result bit 1,if both operand bits are 1; otherwise results bit 0.
    OR: x | y  =14  -> Result bit 1,if any of the operand bit is 1; otherwise results bit 0.
    NOT: ~x    =-11 -> inverts individual bits
    XOR: x ^ y =14  -> Results bit 1,if any of the operand bit is 1 but not both, otherwise results bit 0.
    R-shift: x >> =5 -> As dividing the number by 2
    L-shift: x << =1024 -> As in multiplying the number by 2
    """

    """
    136. Single Number
    Given a non-empty array of integers nums, every element appears twice except for one. 
    Find that single one.
    You must implement a solution with a linear runtime complexity and use only constant extra space.
    Ex1: Input: nums = [2,2,1], Output: 1
    Ex2: Input: nums = [4,1,2,1,2], Output: 4
    Ex3: Input: nums = [1], Output: 1
    """
    """
    Hints: xor
    """
    def singleNumber(self, nums: list[int]) -> int:
        return False

    """
    191. Number of 1 Bits
    Write a function that takes the binary representation of an unsigned integer 
    and returns the number of '1' bits it has (also known as the Hamming weight).
    Therefore, in Example 3, the input represents the signed integer. -3.
    Ex1: Input: n = 00000000000000000000000000001011, Output: 3
    Explanation: The input binary string 00000000000000000000000000001011 
                has a total of three '1' bits.
    Ex2: Input: n = 00000000000000000000000010000000, Output: 1
    Explanation: The input binary string 00000000000000000000000010000000 
                has a total of one '1' bit.
    Ex3: Input: n = 11111111111111111111111111111101, Output: 31
    Explanation: The input binary string 11111111111111111111111111111101 
                has a total of thirty one '1' bits.
    """
    """
    Hints: use bin
    """
    def hammingWeight(self, n: int) -> int:
        return 0

    """
    338. Counting Bits
    Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), 
    ans[i] is the number of 1's in the binary representation of i.
    Ex1: Input: n = 2, Output: [0,1,1]
    Explanation:
        0 --> 0
        1 --> 1
        2 --> 10
    Ex2: Input: n = 5, Output: [0,1,1,2,1,2]
    Explanation:
        0 --> 0   6 --> 110
        1 --> 1   7 --> 111
        2 --> 10  8 --> 1000
        3 --> 11  9 --> 1001
        4 --> 100 10 -->1010 
        5 --> 101
    """
    """
    Hints: use bin
    """
    def countBits(self, n: int) -> list[int]:
        return None

    """
    190. Reverse Bits
    Reverse bits of a given 32 bits unsigned integer.
    Note:
    Note that in some languages, such as Java, there is no unsigned integer type. 
    In this case, both input and output will be given as a signed integer type. 
    They should not affect your implementation, as the integer's internal 
    binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. 
    Therefore, in Example 2 above, the input represents the signed integer -3 and 
    the output represents the signed integer -1073741825.
    Ex1: Input: n = 00000010100101000001111010011100, 
    Output: 964176192 (00111001011110000010100101000000)
    Explanation: The input binary string 00000010100101000001111010011100 
        represents the unsigned integer 43261596, 
        so return 964176192 which its binary representation is 00111001011110000010100101000000.
    Ex2: Input: n = 11111111111111111111111111111101, 
    Output: 3221225471 (10111111111111111111111111111111)
    Explanation: The input binary string 11111111111111111111111111111101 
        represents the unsigned integer 4294967293, 
        so return 3221225471 which its binary representation is 10111111111111111111111111111111.
    """
    """
    Hints: recursive
    """
    def reverseBits(self, n: int) -> int:
        return 0

    """
    268. Missing Number
    Given an array nums containing n distinct numbers in the range [0, n], 
    return the only number in the range that is missing from the array.
    Ex1: Input: nums = [3,0,1], Output: 2
    Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 
        2 is the missing number in the range since it does not appear in nums.
    Ex2: Input: nums = [0,1], Output: 2
    Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 
        2 is the missing number in the range since it does not appear in nums.
    Ex3: Input: nums = [9,6,4,2,3,5,7,0,1], Output: 8
    Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 
        8 is the missing number in the range since it does not appear in nums.
    """
    """
    Hints:
    """
    def missingNumber(self, nums: list[int]) -> int:
        return 0

    """
    371. Sum of Two Integers
    Given two integers a and b, return the sum of the two integers without using the operators + and -.
    Ex1: Input: a = 1, b = 2, Output: 3
    Ex2: Input: a = 2, b = 3, Output: 5
    """
    """
    Hints:
    """
    def getSum(self, a: int, b: int) -> int:
        return 0

    """
    7. Reverse Integer
    Given a signed 32-bit integer x, return x with its digits reversed. 
    If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
    Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
    Ex1: Input: x = 123, Output: 321
    Ex2: Input: x = -123, Output: -321
    Ex3: Input: x = 120, Output: 21
    """
    """
    Hints:
    """
    def reverse(self, x: int) -> int:
        return 0

    # new Picks from 300 list - last 3

    """
    1470. Shuffle the Array
    Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].
    Return the array in the form [x1,y1,x2,y2,...,xn,yn].
    Ex1: Input: nums = [2,5,1,3,4,7], n = 3, Output: [2,3,5,4,1,7] 
        Explanation: Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].
    Ex2: Input: nums = [1,2,3,4,4,3,2,1], n = 4, Output: [1,4,2,3,3,2,4,1]
    Ex3: Input: nums = [1,1,2,2], n = 2, Output: [1,2,1,2]
    """
    """
    Hints:
    """
    def shuffle(self, nums: list[int], n: int) -> list[int]:
        return None

    """
    989. Add to Array-From of Integer
    The array-form of an integer num is an array representing its digits in left to right order.
    For example, for num = 1321, the array form is [1,3,2,1]. Given num, 
    the array-form of an integer, and an integer k, return the array-form of the integer num + k.
    Ex1: Input: num = [1,2,0,0], k = 34, Output: [1,2,3,4], Explanation: 1200 + 34 = 1234
    Ex2: Input: num = [2,7,4], k = 181, Output: [4,5,5], Explanation: 274 + 181 = 455
    Ex3: Input: num = [2,1,5], k = 806, Output: [1,0,2,1], Explanation: 215 + 806 = 1021
    """
    """
    Hints:
    """
    def addToArrayForm(self, num: list[int], k: int) -> list[int]:
        return None

    """
    67. Add Binary
    Given two binary strings a and b, return their sum as a binary string.
    Ex1: Input: a = "11", b = "1", Output: "100"
    Ex2: Input: a = "1010", b = "1011", Output: "10101"
    Constraints:
    1 <= a.length, b.length <= 104
    a and b consist only of '0' or '1' characters.
    Each string does not contain leading zeros except for the zero itself.
    """
    """
    Hints: Use bin
    """
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a, 2) + int(b, 2))[2:]


# if __name__ == '__main__':
#     print("Running contains_duplicate_Test...")
#     nums_1, nums_2, nums_3 = [1, 2, 3, 1], [1, 2, 3, 4], [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
#     if (LeetCode_Practice.contains_duplicate_practice(LeetCode_Practice, nums_1)
#         == LeetCodeNeetCodeProblems.contains_duplicate(nums_1)):
#         print("GOOD JOB BRO")
#     else: print("NOT THE SAME, WRONG CODE for nums_1")
#
#     if (LeetCode_Practice.contains_duplicate_practice(LeetCode_Practice, nums_2)
#         == LeetCodeNeetCodeProblems.contains_duplicate(nums_2)):
#         print("GOOD JOB BRO")
#     else: print("NOT THE SAME, WRONG CODE for nums_2")
#
#     if (LeetCode_Practice.contains_duplicate_practice(LeetCode_Practice, nums_3)
#         == LeetCodeNeetCodeProblems.contains_duplicate(nums_3)):
#         print("GOOD JOB BRO")
#     else: print("NOT THE SAME, WRONG CODE for nums_3")
#
#     print("*****************************************")
#
    # print("Running is_anagram_practice...")
    # s_1, t_1, s_2, t_2 = "anagram", "nagaram", "rat", "car"
    # if (LeetCode_Practice.is_anagram_practice(LeetCode_Practice, s_1, t_1)
    #         == LeetCodeNeetCodeProblems.is_anagram(LeetCodeNeetCodeProblems, s_1, t_1)):
    #     print("GOOD JOB BRO")
    # else:
    #     print("NOT THE SAME, WRONG CODE for nums_1")
    #
    # if (LeetCode_Practice.is_anagram_practice(LeetCode_Practice, s_2, t_2)
    #         == LeetCodeNeetCodeProblems.is_anagram(LeetCodeNeetCodeProblems, s_2, t_2)):
    #     print("GOOD JOB BRO")
    # else:
    #     print("NOT THE SAME, WRONG CODE for nums_2")
#
#     print("*****************************************")
#
#     print("Running two_sum_practice...")
#     nums_1, target_1, nums_2, target_2, nums_3, target_3 = [2,7,11,15], 9, [3,2,4], 6, [3,3], 6
#     if (LeetCode_Practice.two_sum_practice(LeetCode_Practice, nums_1, target_1)
#             == LeetCodeNeetCodeProblems.two_sum(nums_1, target_1)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     if (LeetCode_Practice.two_sum_practice(LeetCode_Practice, nums_2, target_2)
#             == LeetCodeNeetCodeProblems.two_sum(nums_2, target_2)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     if (LeetCode_Practice.two_sum_practice(LeetCode_Practice, nums_3, target_3)
#             == LeetCodeNeetCodeProblems.two_sum(nums_3, target_3)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     print("*****************************************")
#
#     print("Running groupAnagrams_practice...")
#     strs_1, strs_2 = ["eat","tea","tan","ate","nat","bat"], [""]
#     if (LeetCode_Practice.groupAnagrams_practice(LeetCode_Practice, strs_1)
#             == LeetCodeNeetCodeProblems.groupAnagrams(strs_1)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     if (LeetCode_Practice.groupAnagrams_practice(LeetCode_Practice, strs_2)
#             == LeetCodeNeetCodeProblems.groupAnagrams(strs_2)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     print("*****************************************")
#
#     print("Running topKFrequent_practice...")
#     nums_1, k_1, nums_2, k_2 = [1,1,1,2,2,3], 2, [1], 1
#     if (LeetCode_Practice.topKFrequent_practice(LeetCode_Practice, nums_1, k_1)
#             == LeetCodeNeetCodeProblems.topKFrequent(nums_1, k_1)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     if (LeetCode_Practice.topKFrequent_practice(LeetCode_Practice, nums_2, k_2)
#             == LeetCodeNeetCodeProblems.topKFrequent(nums_2, k_2)):
#         print("GOOD JOB BRO")
#     else:
#         print("NOT THE SAME, WRONG CODE for nums_1")
#
#     print("*****************************************")

