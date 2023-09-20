#requirments: python-whois
import whois

#def whois_lookup(url,features):
def whois_lookup(url):
    
    r = whois.whois(url)
    #res = [r.f for f in features]
    return r

if __name__ == "__main__":
    url = input("Enter URL: ")
    print(f"URL:{url}")
    print(whois_lookup(url))
    #print(whois_lookup(url,["creation_date","registrar"]))

