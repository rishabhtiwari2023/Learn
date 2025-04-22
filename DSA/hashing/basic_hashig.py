# 🔐 Intermediate Hashing Questions (DSA)
# ✅ Basic Hashing Patterns
# Count frequency of all elements in an array

# Check if two arrays are equal (unordered)

# Find the first non-repeating character in a string

# Find duplicates in an array

# Count distinct elements in every window of size k

# Check if a pair with given sum exists in an array

# Find the length of the longest subarray with zero sum

# Longest subarray with sum K

# Count subarrays with sum K

# Two Sum problem

# Hashing DSA Intermediate Solutions
# Hashing DSA Intermediate Solutions

# 1. Count frequency of all elements in an array
arr = [1, 2, 2, 3, 1, 4]
freq = {}
for num in arr:
    freq[num] = freq.get(num, 0) + 1

# 2. Check if two arrays are equal (unordered)
arr1 = [1, 2, 2, 3]
arr2 = [2, 1, 3, 2]
freq1, freq2 = {}, {}
for num in arr1:
    freq1[num] = freq1.get(num, 0) + 1
for num in arr2:
    freq2[num] = freq2.get(num, 0) + 1
are_equal = freq1 == freq2

# 3. First non-repeating character in a string
s = "leetcode"
freq = {}
for ch in s:
    freq[ch] = freq.get(ch, 0) + 1
first_index = -1
for i, ch in enumerate(s):
    if freq[ch] == 1:
        first_index = i
        break

# 4. Find duplicates in an array
arr = [1, 2, 3, 2, 3, 4]
freq = {}
duplicates = set()
for num in arr:
    freq[num] = freq.get(num, 0) + 1
    if freq[num] == 2:
        duplicates.add(num)
duplicates = list(duplicates)

# 5. Count distinct elements in every window of size k
arr = [1, 2, 1, 3, 4, 2, 3]
k = 4
result = []
freq = {}
for i in range(len(arr)):
    if i >= k:
        freq[arr[i - k]] -= 1
        if freq[arr[i - k]] == 0:
            del freq[arr[i - k]]
    freq[arr[i]] = freq.get(arr[i], 0) + 1
    if i >= k - 1:
        result.append(len(freq))

# 6. Check if a pair with given sum exists
arr = [1, 4, 7, 2, 9]
target = 6
seen = set()
pair_exists = False
for num in arr:
    if target - num in seen:
        pair_exists = True
        break
    seen.add(num)

# 7. Longest subarray with zero sum
arr = [1, 2, -3, 3, -1, -2, 4]
prefix_sum = 0
seen = {0: -1}
max_len = 0
for i, num in enumerate(arr):
    prefix_sum += num
    if prefix_sum in seen:
        max_len = max(max_len, i - seen[prefix_sum])
    else:
        seen[prefix_sum] = i

# 8. Longest subarray with sum K
arr = [1, 2, 3, -2, 5]
k = 5
prefix_sum = 0
seen = {0: -1}
max_len = 0
for i, num in enumerate(arr):
    prefix_sum += num
    if prefix_sum - k in seen:
        max_len = max(max_len, i - seen[prefix_sum - k])
    if prefix_sum not in seen:
        seen[prefix_sum] = i

# 9. Count subarrays with sum K
arr = [1, 1, 1]
k = 2
prefix_sum = 0
freq = {0: 1}
count = 0
for num in arr:
    prefix_sum += num
    count += freq.get(prefix_sum - k, 0)
    freq[prefix_sum] = freq.get(prefix_sum, 0) + 1

# 10. Two Sum
nums = [2, 7, 11, 15]
target = 9
index_map = {}
result = []
for i, num in enumerate(nums):
    complement = target - num
    if complement in index_map:
        result = [index_map[complement], i]
        break
    index_map[num] = i


# 🔁 String and HashMap
# Group anagrams together

# Check if two strings are isomorphic

# Check if a string can be rearranged to form a palindrome

# Longest substring without repeating characters

# Longest repeating character replacement (Leetcode 424)

# Longest substring with at most K distinct characters

# Find the index of the first unique character in a string

# Check if a string is a permutation of another string

# Minimum window substring


# 11. Group anagrams together
def group_anagrams(strs):
    anagram_map = {}
    for word in strs:
        key = ''.join(sorted(word))
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(word)
    return list(anagram_map.values())


# 12. Check if two strings are isomorphic
def are_isomorphic(s, t):
    if len(s) != len(t):
        return False
    s_map, t_map = {}, {}
    for i in range(len(s)):
        if s[i] in s_map and s_map[s[i]] != t[i]:
            return False
        if t[i] in t_map and t_map[t[i]] != s[i]:
            return False
        s_map[s[i]] = t[i]
        t_map[t[i]] = s[i]
    return True

# 13. Check if a string can be rearranged to form a palindrome
def can_form_palindrome(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    odd_count = 0
    for val in freq.values():
        if val % 2 != 0:
            odd_count += 1
    return odd_count <= 1


# 14. Longest substring without repeating characters
def longest_unique_substring(s):
    seen = {}
    left = max_len = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len

# 15. Longest repeating character replacement
def character_replacement(s, k):
    count = {}
    left = max_freq = result = 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_freq = max(max_freq, count[s[right]])
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        result = max(result, right - left + 1)
    return result


# 16. Longest substring with at most K distinct characters
def length_of_longest_k_substring(s, k):
    char_map = {}
    left = max_len = 0
    for right, ch in enumerate(s):
        char_map[ch] = char_map.get(ch, 0) + 1
        while len(char_map) > k:
            char_map[s[left]] -= 1
            if char_map[s[left]] == 0:
                del char_map[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 17. Find the index of the first unique character in a string
def first_uniq_char(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    for i in range(len(s)):
        if freq[s[i]] == 1:
            return i
    return -1


# 18. Check if a string is a permutation of another string
def is_permutation(s1, s2):
    if len(s1) != len(s2):
        return False
    freq1 = {}
    freq2 = {}
    for ch in s1:
        freq1[ch] = freq1.get(ch, 0) + 1
    for ch in s2:
        freq2[ch] = freq2.get(ch, 0) + 1
    return freq1 == freq2


# 19. Minimum window substring
def min_window(s, t):
    if not s or not t:
        return ""
    dict_t = {}
    for ch in t:
        dict_t[ch] = dict_t.get(ch, 0) + 1

    required = len(dict_t)
    window_counts = {}
    l = r = formed = 0
    ans = float("inf"), None, None

    while r < len(s):
        ch = s[r]
        window_counts[ch] = window_counts.get(ch, 0) + 1

        if ch in dict_t and window_counts[ch] == dict_t[ch]:
            formed += 1

        while l <= r and formed == required:
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            ch = s[l]
            window_counts[ch] -= 1
            if ch in dict_t and window_counts[ch] < dict_t[ch]:
                formed -= 1
            l += 1
        r += 1

    return "" if ans[0] == float("inf") else s[ans[1]:ans[2]+1]


# Rabin-Karp string matching using hashing

# 🔄 Prefix & Subarray Hashing
# Subarray sum divisible by K

# Count number of nice subarrays (Leetcode 1248)

# Count of subarrays with equal number of 0s and 1s

# Number of submatrices that sum to target

# Count of zero sum subarrays in 2D matrix

# Count subarrays with product less than K using hashing

# 📦 Set & Unordered Map Patterns
# Longest consecutive sequence in an array

# Find missing and repeating numbers using hashing

# Find all pairs with given XOR

# Intersection of two arrays


# 20. Rabin-Karp string matching using hashing
def rabin_karp(text, pattern, q=101):
    d = 256
    M = len(pattern)
    N = len(text)
    p = 0
    t = 0
    h = 1
    results = []

    for i in range(M-1):
        h = (h*d)%q

    for i in range(M):
        p = (d*p + ord(pattern[i]))%q
        t = (d*t + ord(text[i]))%q

    for i in range(N-M+1):
        if p == t:
            if text[i:i+M] == pattern:
                results.append(i)
        if i < N-M:
            t = (d*(t - ord(text[i])*h) + ord(text[i+M]))%q
            if t < 0:
                t += q

    return results

# 21. Subarray sum divisible by K
def number_of_nice_subarrays(nums, k):
    freq = {}
    freq[0] = 1
    count = 0
    prefix_sum = 0

    for num in nums:
        prefix_sum += num % 2  # count odd numbers so far
        if (prefix_sum - k) in freq:
            count += freq[prefix_sum - k]
        freq[prefix_sum] = freq.get(prefix_sum, 0) + 1

    return count


# 22. Count number of nice subarrays
def num_submatrix_sum_target(matrix, target):
    rows, cols = len(matrix), len(matrix[0])

    # Precompute prefix sum row-wise
    for r in range(rows):
        for c in range(1, cols):
            matrix[r][c] += matrix[r][c - 1]

    count = 0

    for c1 in range(cols):
        for c2 in range(c1, cols):
            sums = {}
            sums[0] = 1
            curr_sum = 0

            for r in range(rows):
                val = matrix[r][c2]
                if c1 > 0:
                    val -= matrix[r][c1 - 1]
                curr_sum += val
                if (curr_sum - target) in sums:
                    count += sums[curr_sum - target]
                sums[curr_sum] = sums.get(curr_sum, 0) + 1

    return count

# 23. Count of subarrays with equal number of 0s and 1s
def count_subarrays_equal_0_1(arr):
    for i in range(len(arr)):
        arr[i] = -1 if arr[i] == 0 else 1
    return count_subarrays_sum_k(arr, 0)

# 24. Number of submatrices that sum to target
def num_submatrix_sum_target(matrix, target):
    from collections import defaultdict
    rows, cols = len(matrix), len(matrix[0])
    for r in range(rows):
        for c in range(1, cols):
            matrix[r][c] += matrix[r][c-1]

    count = 0
    for c1 in range(cols):
        for c2 in range(c1, cols):
            sums = defaultdict(int)
            curr_sum = 0
            sums[0] = 1
            for r in range(rows):
                val = matrix[r][c2] - (matrix[r][c1-1] if c1 > 0 else 0)
                curr_sum += val
                count += sums[curr_sum - target]
                sums[curr_sum] += 1
    return count

# 25. Count of zero sum subarrays in 2D matrix (reuse 24 with target = 0)

# 26. Count subarrays with product less than K (using sliding window, not hashing)
def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0
    prod = 1
    left = 0
    result = 0
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod //= nums[left]
            left += 1
        result += right - left + 1
    return result

# 27. Longest consecutive sequence
def longest_consecutive(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            longest = max(longest, length)
    return longest

# 28. Find missing and repeating numbers
def find_missing_and_repeating(arr):
    n = len(arr)
    freq = Counter(arr)
    for i in range(1, n+1):
        if freq[i] == 0:
            missing = i
        elif freq[i] > 1:
            repeating = i
    return (repeating, missing)

# 29. Find all pairs with given XOR
def find_pairs_with_xor(arr, x):
    seen = set()
    pairs = []
    for num in arr:
        if num ^ x in seen:
            pairs.append((num, num ^ x))
        seen.add(num)
    return pairs

# 30. Intersection of two arrays
def intersection(arr1, arr2):
    return list(set(arr1) & set(arr2))


# Union of two unsorted arrays

# Find elements appearing more than ⌊n/3⌋ times

# 💥 Advanced Use Cases
# Detect cycle in a linked list using hashing

# Clone a linked list with random pointers using hashing

# Detect duplicate subtrees in a binary tree

# Count paths with sum K in a binary tree

# Serialize and deserialize a binary tree using hashmap

# Smallest subarray with all occurrences of the most frequent element

# Count pairs with given XOR using hashing

# Maximum frequency stack (Leetcode 895)

# Hashing DSA Intermediate Solutions

# 30. Intersection of two arrays
def intersection(arr1, arr2):
    return list(set(arr1) & set(arr2))

# 31. Union of two arrays
def union(arr1, arr2):
    return list(set(arr1) | set(arr2))

# 32. Check if an array can be divided into pairs whose sum is divisible by k
def can_pair(arr, k):
    from collections import defaultdict
    freq = defaultdict(int)
    for num in arr:
        freq[num % k] += 1
    for rem in freq:
        if rem == 0:
            if freq[rem] % 2 != 0:
                return False
        elif freq[rem] != freq[k - rem]:
            return False
    return True

# 33. Group Anagrams
def group_anagrams(strs):
    from collections import defaultdict
    anagrams = defaultdict(list)
    for word in strs:
        key = tuple(sorted(word))
        anagrams[key].append(word)
    return list(anagrams.values())

# 34. Top K frequent elements
def top_k_frequent(nums, k):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:k]]

# 35. Sort characters by frequency
def frequency_sort(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    sorted_chars = sorted(freq, key=lambda ch: freq[ch], reverse=True)
    return ''.join([ch * freq[ch] for ch in sorted_chars])

# 36. Find the number of good pairs (i, j) where nums[i] == nums[j] and i < j
def num_identical_pairs(nums):
    from collections import Counter
    freq = Counter(nums)
    return sum((v * (v - 1)) // 2 for v in freq.values())

# 37. Maximum number of non-overlapping subarrays with sum equal to target
def max_non_overlapping(nums, target):
    prefix_sum = 0
    seen = {0}
    count = 0
    for num in nums:
        prefix_sum += num
        if prefix_sum - target in seen:
            count += 1
            seen = {0}
            prefix_sum = 0
        else:
            seen.add(prefix_sum)
    return count

# 38. Find common elements in three sorted arrays
def common_elements(arr1, arr2, arr3):
    return list(set(arr1) & set(arr2) & set(arr3))

# 39. Longest substring without repeating characters
def length_of_longest_substring(s):
    seen = {}
    start = max_len = 0
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len

# 40. Find all anagrams of a pattern in a string
def find_anagrams(s, p):
    res = []
    p_count = {}
    s_count = {}

    for ch in p:
        p_count[ch] = p_count.get(ch, 0) + 1

    for ch in s[:len(p) - 1]:
        s_count[ch] = s_count.get(ch, 0) + 1

    for i in range(len(p) - 1, len(s)):
        ch = s[i]
        s_count[ch] = s_count.get(ch, 0) + 1
        if s_count == p_count:
            res.append(i - len(p) + 1)
        left_char = s[i - len(p) + 1]
        s_count[left_char] -= 1
        if s_count[left_char] == 0:
            del s_count[left_char]
    return res
