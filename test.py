from fuzzywuzzy import fuzz
from fuzzywuzzy import process

print(fuzz.token_sort_ratio("in my opinion", "in opinion my honest"))
print(fuzz.token_sort_ratio("in my opinion", "in my honest opinion"))
print(fuzz.token_sort_ratio("in my opinion", "opinion in my"))
print(fuzz.token_sort_ratio("in my opinion", "opinion"))