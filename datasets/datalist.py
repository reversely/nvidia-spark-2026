import requests

url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_list"
resp = requests.get(url)
names = resp.json()["result"]

with open("datasets.txt", "w") as f:
    f.write("\n".join(names) + "\n")

print(f"Saved {len(names)} datasets to datasets.txt")
    