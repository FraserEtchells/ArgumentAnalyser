from fuzzywuzzy import fuzz
from fuzzywuzzy import process

precision = 0.8104
recall = 0.8115

f1 = 2 * ((precision * recall) / (precision + recall))
print(f1)