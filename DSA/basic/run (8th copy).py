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
    print(freq)
    if i >= k - 1:
        result.append(len(freq))
print(result)