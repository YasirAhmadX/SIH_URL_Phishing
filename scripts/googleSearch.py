try:
	from googlesearch import search
except ImportError:
	print("No module named 'google' found")

url = input()
# to search
query = "site:" + url

for j in search(query, tld="co.in", num=10, stop=10, pause=2):
	print(j)

try:
    j = j
    print("seems legit")
except NameError:
    print("Not listed google")
