# 1. Fibonacci Sequence
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(list(fibonacci(10)))

# 2. Palindrome Check
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("racecar"))
print(is_palindrome("hello"))

# 3. Anagram Check
def are_anagrams(str1, str2):
    return sorted(str1) == sorted(str2)

print(are_anagrams("listen", "silent"))
print(are_anagrams("hello", "world"))

# 4. Prime Number Check
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i != 0:
            return False
    return True

print(is_prime(11))
print(is_prime(4))

# 5. Merge Two Sorted Lists
def merge_sorted_lists(list1, list2):
    sorted_list = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            sorted_list.append(list1[i])
            i += 1
        else:
            sorted_list.append(list2[j])
            j += 1
    sorted_list.extend(list1[i:])
    sorted_list.extend(list2[j:])
    return sorted_list

print(merge_sorted_lists([1, 3, 5], [2, 4, 6]))

# 6. Find the Missing Number in an Array
def find_missing_number(arr, n):
    return n * (n + 1) // 2 - sum(arr)

print(find_missing_number([1, 2, 4, 5, 6], 6))

# 7. Longest Common Subsequence
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

print(longest_common_subsequence("abcde", "ace"))

# 8. Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

print(binary_search([1, 2, 3, 4, 5], 3))
print(binary_search([1, 2, 3, 4, 5], 6))

# 9. Find Duplicates in an Array
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for num in arr:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    return list(duplicates)

print(find_duplicates([1, 2, 3, 4, 4, 5, 5, 6]))

# 10. Reverse a Linked List
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

# Helper function to print linked list
def print_linked_list(head):
    while head:
        print(head.value, end=" -> ")
        head = head.next
    print("None")

# Create a linked list 1 -> 2 -> 3 -> None
head = ListNode(1, ListNode(2, ListNode(3)))
print_linked_list(head)
reversed_head = reverse_linked_list(head)
print_linked_list(reversed_head)


# 11. Find the Largest Element in an Array
def find_largest(arr):
    return max(arr)

print(find_largest([1, 2, 3, 4, 5]))

# 12. Find the Second Largest Element in an Array
def find_second_largest(arr):
    unique_arr = list(set(arr))
    unique_arr.sort()
    return unique_arr[-2]

print(find_second_largest([1, 2, 3, 4, 5]))

# 13. Find the Intersection of Two Arrays
def find_intersection(arr1, arr2):
    return list(set(arr1) & set(arr2))

print(find_intersection([1, 2, 3], [2, 3, 4]))

# 14. Find the Union of Two Arrays
def find_union(arr1, arr2):
    return list(set(arr1) | set(arr2))

print(find_union([1, 2, 3], [2, 3, 4]))

# 15. Find the Duplicates in an Array
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for num in arr:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    return list(duplicates)

print(find_duplicates([1, 2, 3, 4, 4, 5, 5, 6]))

# 16. Find the Most Frequent Element in an Array
def find_most_frequent(arr):
    return max(set(arr), key=arr.count)

print(find_most_frequent([1, 2, 3, 4, 4, 5, 5, 6]))

# 17. Find the Longest Palindromic Substring
def longest_palindromic_substring(s):
    n = len(s)
    if n == 0:
        return ""
    result = s[0]
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]
            if substring == substring[::-1] and len(substring) > len(result):
                result = substring
    return result

print(longest_palindromic_substring("babad"))

# 18. Find the Longest Increasing Subsequence
def longest_increasing_subsequence(arr):
    if not arr:
        return []
    lis = [arr[0]]
    for num in arr[1:]:
        if num > lis[-1]:
            lis.append(num)
        else:
            for i in range(len(lis)):
                if lis[i] >= num:
                    lis[i] = num
                    break
    return lis

print(longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]))

# 19. Find the Longest Common Prefix
def longest_common_prefix(strs):
    if not strs:
        return ""
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for other in strs:
            if other[i] != char:
                return shortest[:i]
    return shortest

print(longest_common_prefix(["flower", "flow", "flight"]))

# 20. Find the Minimum in a Rotated Sorted Array
def find_min_in_rotated_sorted_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]

print(find_min_in_rotated_sorted_array([4, 5, 6, 7, 0, 1, 2]))

# 21. Find the Peak Element in an Array
def find_peak_element(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left

print(find_peak_element([1, 2, 3, 1]))

# 22. Find the Majority Element in an Array
def find_majority_element(arr):
    count = 0
    candidate = None
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate

print(find_majority_element([3, 2, 3]))

# 23. Find the Missing Ranges in an Array
def find_missing_ranges(arr, lower, upper):
    result = []
    prev = lower - 1
    for num in arr + [upper + 1]:
        if num == prev + 2:
            result.append(str(prev + 1))
        elif num > prev + 2:
            result.append(f"{prev + 1}->{num - 1}")
        prev = num
    return result

print(find_missing_ranges([0, 1, 3, 50, 75], 0, 99))

# 24. Find the Kth Largest Element in an Array
def find_kth_largest(arr, k):
    return sorted(arr, reverse=True)[k - 1]

print(find_kth_largest([3, 2, 1, 5, 6, 4], 2))

# 25. Find the Kth Smallest Element in an Array
def find_kth_smallest(arr, k):
    return sorted(arr)[k - 1]

print(find_kth_smallest([3, 2, 1, 5, 6, 4], 2))

# 26. Find the Maximum Subarray Sum
def max_subarray_sum(arr):
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

print(max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]))

# 27. Find the Minimum Subarray Sum
def min_subarray_sum(arr):
    min_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = min(num, current_sum + num)
        min_sum = min(min_sum, current_sum)
    return min_sum

print(min_subarray_sum([2,1,3,4,1,2,1,5,4]))

# 28. Find the Maximum Product Subarray
def max_product_subarray(arr):
    max_product = min_product = result = arr[0]
    for num in arr[1:]:
        if num < 0:
            max_product, min_product = min_product, max_product
        max_product = max(num, max_product * num)
        min_product = min(num, min_product * num)
        result = max(result, max_product)
    return result

print(max_product_subarray([2,3,-2,4]))

# 29. Find the Minimum Product Subarray
def min_product_subarray(arr):
    max_product = min_product = result = arr[0]
    for num in arr[1:]:
        if num < 0:
            max_product, min_product = min_product, max_product
        max_product = max(num, max_product * num)
        min_product = min(num, min_product * num)
        result = min(result, min_product)
    return result

print(min_product_subarray([2,3,-2,4]))

# 30. Find the Maximum Sum Increasing Subsequence
def max_sum_increasing_subsequence(arr):
    n = len(arr)
    max_sum = arr[:]
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j] and max_sum[i] < max_sum[j] + arr[i]:
                max_sum[i] = max_sum[j] + arr[i]
    return max(max_sum)

print(max_sum_increasing_subsequence([1, 101, 2, 3, 100, 4, 5]))



# 31. String Compression
def compress_string(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s) + 1):
        if i < len(s) and s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1
    return ''.join(result)

print(compress_string("aabcccccaaa"))  # Output: "a2b1c5a3"

# 32. Running Sum of 1D Array
def running_sum(nums):
    result = []
    current_sum = 0
    for num in nums:
        current_sum += num
        result.append(current_sum)
    return result

print(running_sum([1, 2, 3, 4]))  # Output: [1, 3, 6, 10]

# 33. Valid Parentheses
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:  # closing bracket
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:  # opening bracket
            stack.append(char)
    return not stack

print(is_valid_parentheses("()[]{}"))  # Output: True
print(is_valid_parentheses("([)]"))    # Output: False

# 34. Count Unique Words in Text
def count_unique_words(text):
    words = text.lower().split()
    # Remove punctuation from each word
    words = [word.strip('.,!?;:()[]{}""\'') for word in words]
    return len(set(words))

print(count_unique_words("The quick brown fox jumps over the lazy dog."))

# 35. Generate Pascal's Triangle
def generate_pascal_triangle(num_rows):
    result = []
    for i in range(num_rows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = result[i-1][j-1] + result[i-1][j]
        result.append(row)
    return result

print(generate_pascal_triangle(5))

# 36. Check if Number is Power of Two
def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

print(is_power_of_two(16))  # Output: True
print(is_power_of_two(18))  # Output: False

# 37. Count Bits in a Number
def count_bits(n):
    return bin(n).count('1')

print(count_bits(7))   # Output: 3 (111 in binary)
print(count_bits(10))  # Output: 2 (1010 in binary)

# 38. Valid Anagram with Counter
from collections import Counter

def is_anagram(s, t):
    return Counter(s) == Counter(t)

print(is_anagram("anagram", "nagaram"))  # Output: True
print(is_anagram("rat", "car"))          # Output: False

# 39. Rotate Array
def rotate_array(nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]
    return nums

print(rotate_array([1, 2, 3, 4, 5, 6, 7], 3))  # Output: [5, 6, 7, 1, 2, 3, 4]

# 40. Flatten Nested List
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

print(flatten_list([1, [2, [3, 4], 5], 6]))  # Output: [1, 2, 3, 4, 5, 6]

# 41. Calculate Moving Average
def moving_average(nums, window_size):
    results = []
    window_sum = sum(nums[:window_size])
    results.append(window_sum / window_size)
    
    for i in range(window_size, len(nums)):
        window_sum = window_sum - nums[i - window_size] + nums[i]
        results.append(window_sum / window_size)
    
    return results

print(moving_average([1, 3, 5, 7, 9], 3))  # Output: [3.0, 5.0, 7.0]

# 42. Matrix Traversal (Spiral Order)
def spiral_order(matrix):
    if not matrix:
        return []
        
    result = []
    rows, cols = len(matrix), len(matrix[0])
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        if top <= bottom:
            # Traverse left
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        if left <= right:
            # Traverse up
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
            
    return result

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(spiral_order(matrix))  # Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]

# 43. Group Anagrams
def group_anagrams(strs):
    anagram_groups = {}
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    return list(anagram_groups.values())

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

# 44. Find First and Last Position in Sorted Array
def search_range(nums, target):
    def binary_search(find_first):
        left, right = 0, len(nums) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                result = mid
                if find_first:
                    right = mid - 1  # Continue searching on the left
                else:
                    left = mid + 1   # Continue searching on the right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    
    first = binary_search(True)
    if first == -1:
        return [-1, -1]
    last = binary_search(False)
    return [first, last]

print(search_range([5,7,7,8,8,10], 8))  # Output: [3, 4]

# 45. Convert Temperature
def convert_temperature(celsius):
    # Convert to Fahrenheit and Kelvin
    fahrenheit = celsius * 9/5 + 32
    kelvin = celsius + 273.15
    return [kelvin, fahrenheit]

print(convert_temperature(36.50))  # Output: [309.65, 97.7]

# 46. Remove Duplicates from Sorted Array In-Place
def remove_duplicates(nums):
    if not nums:
        return 0
    
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    
    return i + 1  # Return length of the array without duplicates

nums = [1, 1, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(nums)
print(nums[:length])  # Output: [1, 2, 3, 4, 5]

# 47. Jump Game
def can_jump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    return True

print(can_jump([2,3,1,1,4]))  # Output: True
print(can_jump([3,2,1,0,4]))  # Output: False

# 48. Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    char_dict = {}
    max_length = start = 0
    
    for i, char in enumerate(s):
        if char in char_dict and start <= char_dict[char]:
            start = char_dict[char] + 1
        else:
            max_length = max(max_length, i - start + 1)
        char_dict[char] = i
    
    return max_length

print(length_of_longest_substring("abcabcbb"))  # Output: 3
print(length_of_longest_substring("pwwkew"))    # Output: 3

# 49. Valid Sudoku
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            if board[i][j] == ".":
                continue
                
            num = board[i][j]
            box_idx = (i // 3) * 3 + j // 3
            
            if (num in rows[i] or num in cols[j] or num in boxes[box_idx]):
                return False
            
            rows[i].add(num)
            cols[j].add(num)
            boxes[box_idx].add(num)
    
    return True

# 50. Implement Queue using Stacks
class MyQueue:
    def __init__(self):
        self.stack1 = []  # for push
        self.stack2 = []  # for pop
        
    def push(self, x):
        self.stack1.append(x)
        
    def pop(self):
        self._move_elements_if_needed()
        return self.stack2.pop()
        
    def peek(self):
        self._move_elements_if_needed()
        return self.stack2[-1]
        
    def empty(self):
        return not self.stack1 and not self.stack2
    
    def _move_elements_if_needed(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

# Test the queue
queue = MyQueue()
queue.push(1)
queue.push(2)
print(queue.peek())  # Output: 1
print(queue.pop())   # Output: 1
print(queue.empty()) # Output: False



# =========================================================================================
# 51. Check if a Number is Even or Odd
def is_even(num):
    return num % 2 == 0

print(is_even(4))  # Output: True
print(is_even(7))  # Output: False

# 52. Find the Factorial of a Number
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

print(factorial(5))  # Output: 120

# 53. Reverse a String Word by Word
def reverse_words(s):
    words = s.split()
    return ' '.join(words[::-1])

print(reverse_words("Hello World"))  # Output: "World Hello"

# 54. Count Vowels in a String
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

print(count_vowels("Hello World"))  # Output: 3

# 55. Check if a String Contains Only Digits
def is_digit_string(s):
    return s.isdigit()

print(is_digit_string("12345"))  # Output: True
print(is_digit_string("123a45"))  # Output: False

# 56. Remove Specific Element from a List
def remove_element(arr, val):
    return [item for item in arr if item != val]

print(remove_element([1, 2, 3, 4, 2], 2))  # Output: [1, 3, 4]

# 57. Sum of Even Numbers in a List
def sum_even_numbers(arr):
    return sum(num for num in arr if num % 2 == 0)

print(sum_even_numbers([1, 2, 3, 4, 5, 6]))  # Output: 12

# 58. Check if a Number is a Perfect Square
def is_perfect_square(num):
    sqrt = int(num ** 0.5)
    return sqrt * sqrt == num

print(is_perfect_square(16))  # Output: True
print(is_perfect_square(15))  # Output: False

# 59. Sum of Digits in a Number
def sum_of_digits(num):
    return sum(int(digit) for digit in str(num))

print(sum_of_digits(123))  # Output: 6

# 60. Find the GCD (Greatest Common Divisor) of Two Numbers
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

print(gcd(48, 18))  # Output: 6

# 61. Find the LCM (Least Common Multiple) of Two Numbers
def lcm(a, b):
    return a * b // gcd(a, b)

print(lcm(12, 15))  # Output: 60

# 62. Check if a String Starts with a Specific Substring
def starts_with(s, prefix):
    return s.startswith(prefix)

print(starts_with("Hello World", "Hello"))  # Output: True
print(starts_with("Hello World", "World"))  # Output: False

# 63. Count Occurrences of a Character in a String
def count_char(s, char):
    return s.count(char)

print(count_char("programming", "m"))  # Output: 2

# 64. Find the Index of the First Occurrence of a Substring
def find_substring(s, sub):
    return s.find(sub)

print(find_substring("Hello World", "World"))  # Output: 6
print(find_substring("Hello World", "Planet"))  # Output: -1

# 65. Convert a String to Uppercase
def to_uppercase(s):
    return s.upper()

print(to_uppercase("hello"))  # Output: "HELLO"

# 66. Convert a String to Lowercase
def to_lowercase(s):
    return s.lower()

print(to_lowercase("HELLO"))  # Output: "hello"

# 67. Check if All Elements in a List Are Equal
def all_equal(arr):
    return all(elem == arr[0] for elem in arr)

print(all_equal([1, 1, 1, 1]))  # Output: True
print(all_equal([1, 2, 1, 1]))  # Output: False

# 68. Find the Average of List Elements
def average(arr):
    if not arr:
        return 0
    return sum(arr) / len(arr)

print(average([1, 2, 3, 4, 5]))  # Output: 3.0

# 69. Remove Whitespace from a String
def remove_whitespace(s):
    return ''.join(s.split())

print(remove_whitespace("Hello World"))  # Output: "HelloWorld"

# 70. Check if a String is Alphanumeric
def is_alphanumeric(s):
    return s.isalnum()

print(is_alphanumeric("abc123"))  # Output: True
print(is_alphanumeric("abc 123"))  # Output: False

# 71. Sort a List in Ascending Order
def sort_list(arr):
    return sorted(arr)

print(sort_list([3, 1, 4, 1, 5, 9, 2]))  # Output: [1, 1, 2, 3, 4, 5, 9]

# 72. Sort a List in Descending Order
def sort_list_desc(arr):
    return sorted(arr, reverse=True)

print(sort_list_desc([3, 1, 4, 1, 5, 9, 2]))  # Output: [9, 5, 4, 3, 2, 1, 1]

# 73. Find Common Elements Between Two Lists
def common_elements(list1, list2):
    return list(set(list1) & set(list2))

print(common_elements([1, 2, 3, 4], [3, 4, 5, 6]))  # Output: [3, 4]

# 74. Generate a List of Random Numbers
import random
def generate_random_list(n, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(n)]

print(generate_random_list(5, 1, 10))  # Output: Random list of 5 numbers between 1 and 10

# 75. Check if a String is a Pangram (Contains All Letters of Alphabet)
def is_pangram(s):
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    return set(s.lower()) >= alphabet

print(is_pangram("The quick brown fox jumps over the lazy dog"))  # Output: True
print(is_pangram("Hello world"))  # Output: False

# 76. Convert a List of Integers to a List of Strings
def int_to_str_list(arr):
    return [str(num) for num in arr]

print(int_to_str_list([1, 2, 3, 4, 5]))  # Output: ['1', '2', '3', '4', '5']

# 77. Count the Number of Words in a String
def count_words(s):
    return len(s.split())

print(count_words("Hello world, how are you?"))  # Output: 5

# 78. Check if a String is a Valid Email Address
import re
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

print(is_valid_email("example@email.com"))  # Output: True
print(is_valid_email("invalid-email"))  # Output: False

# 79. Remove Duplicates from a List While Preserving Order
def remove_duplicates_preserve_order(arr):
    seen = set()
    return [x for x in arr if not (x in seen or seen.add(x))]

print(remove_duplicates_preserve_order([1, 2, 3, 1, 2, 4, 5]))  # Output: [1, 2, 3, 4, 5]

# 80. Generate a Sequence of Tribonacci Numbers
def tribonacci(n):
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    if n == 3:
        return [0, 1, 1]
    
    trib = [0, 1, 1]
    for i in range(3, n):
        trib.append(trib[i-1] + trib[i-2] + trib[i-3])
    return trib

print(tribonacci(10))  # Output: [0, 1, 1, 2, 4, 7, 13, 24, 44, 81]
