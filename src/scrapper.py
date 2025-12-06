import requests
from requests import Response
from bs4 import BeautifulSoup
from typing import Any
import os
import csv
import time
import random
from urls import urlsA, urlsB, urlsC
import re

def normalize_name(name: str) -> str:
  name = name.strip()
  name = re.sub(r"\(.*?\)", "", name).strip()
  if "," in name:
      parts = [p.strip() for p in name.split(",")]
      name = parts[-1]  # última parte
  name = re.sub(r"^Swarm of\s+", "", name, flags=re.I)
  name = name.replace(" ", "_")
  name = "_".join(word.capitalize() for word in name.split("_"))

  return name

def load_names(filepath: str):
  normalized_names = set()

  with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
          continue
      name = normalize_name(line)
      normalized_names.add(name)

  name_list = list(normalized_names)
  name_list.sort()
  return name_list


def log_error(result: dict):
  with open("errors2.txt", "a", encoding="utf-8") as f:
    f.write(
      f"{result['url']} - {result['message']}\n"
    )

def save_success_to_csv(monster_name: str, description: str):
  file_exists = os.path.isfile("output2.csv")

  with open("output2.csv", "a", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

    if not file_exists:
      writer.writerow(["monster_name", "description"])
    cleaned_description = description.strip().replace('\n', ' ').replace('\r', ' ')
    writer.writerow([monster_name, cleaned_description])

def extract_description(url: str) -> dict[str, Any]:
    try:
      response: Response = requests.get(url, timeout=10)
      response.raise_for_status()
    except requests.exceptions.RequestException as e:
      return {
        "url": url,
        "success": False,
        "error_code": "REQUEST_ERROR",
        "message": str(e)
      }

    soup = BeautifulSoup(response.text, "html.parser")

    monster_name = None
    for h1 in soup.find_all("h1"):
      span = h1.find("span")
      if span:
        monster_name = span.get_text(strip=True).lower() 

    description_h2 = None
    for h2 in soup.find_all("h2"):
        span = h2.find("span")
        if span and span.get_text(strip=True).lower() == "description":
            description_h2 = h2
            break

    if not description_h2:
        return {
        "url": url,
        "success": False,
        "error_code": "REQUEST_ERROR",
        "message": "DESCRIPTION NOT PRESENT"
      }

    content = []
    for sibling in description_h2.find_all_next():
        if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            break
        if sibling.name == "p":
            content.append(sibling.get_text(strip=True))

    if not content:
      return {
        "url": url,
        "error_code": "EMPTY_SECTION",
        "success": False,
        "message": "No <p> elements found after the Description section."
      }

    description = "\n".join(content)
    return {
      "success": True,
      "monster_name": monster_name,
      "description": description
      }


if __name__ == "__main__":
  # urls = [
  #   "https://forgottenrealms.fandom.com/wiki/Kenkus",
  # ]

  file_path = "monster_names.txt"
  name_list = load_names(file_path)
  print(len(name_list))
  print(name_list)
  
  current_initial = "A"
 
  for name in name_list:
    title_name= name.lower()
    title_name = title_name[0].upper() + title_name[1:]
    
    if title_name[0] != current_initial:
      current_initial = title_name[0]
      random.uniform(10, 15)
    
    url = f"https://forgottenrealms.fandom.com/wiki/{title_name}"
    print(f"\n--- URL: {url} ---")
    time.sleep(random.uniform(1, 5))
    res = extract_description(url)
    
    if (res.get("success") == False):
      log_error(res)
    else:
      save_success_to_csv(res["monster_name"], res["description"])
        