import requests
from requests import Response
from bs4 import BeautifulSoup
from typing import Any, List, Dict
import os
import csv
import time
import random
import re
from pathlib import Path

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


def log_error(result: dict, error_file:str):
  with open(error_file, "a", encoding="utf-8") as f:
    f.write(
      f"{result['url']} - {result['message']}\n"
    )

def save_success_to_csv(dict_to_save: Dict[str, str], output_file: str | Path):
  
  file_exists = os.path.isfile(output_file)

  with open(output_file, "a", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

    if not file_exists:
      writer.writerow(dict_to_save.keys())
    for key in dict_to_save.keys():
      dict_to_save[key] = dict_to_save[key].strip().replace('\n', ' ').replace('\r', ' ')
    writer.writerow(dict_to_save.values())

def find_content(title:str, title_tag: str, soup: BeautifulSoup):
  content_title = None
  for t_tag in soup.find_all(title_tag):
    span = t_tag.find("span")
    if span and (span.get_text(strip=True).lower() == title.lower() or "type" in span.get_text(strip=True)):
      content_title = t_tag
      break
  if not content_title:
    return "NOT FOUND"
  else:
    content = []
    for sibling in content_title.find_all_next():
      if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        break
      if sibling.name == "p":
        content.append(sibling.get_text(strip=True))
    return "\n".join(content) 

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

    monster_name = "Name Not Found"
    for h1 in soup.find_all("h1"):
      span = h1.find("span")
      if span:
        monster_name = span.get_text(strip=True).lower() 

    abstract = find_content(monster_name, "h1", soup)
    description = find_content("description", "h2", soup)
    personality = find_content("personality", "h2", soup)
    realm = find_content("realm", "h2", soup)
    activities = find_content("activities", "h2", soup)
    combat = find_content("combat", "h2", soup)
    biology = find_content("biology", "h2", soup)
    society = find_content("society", "h2", soup)
    abilities = find_content("abilities", "h2", soup)
    behavior = find_content("behavior", "h2", soup)
    reputation = find_content("reputation", "h2", soup)
    culture = find_content("culture", "h3", soup)
    sub_races = find_content("sub-races", "h3", soup)
    types = find_content("types", "h2", soup)
    history = find_content("history", "h2", soup)
    rumors = find_content("rumors & legends", "h2", soup)
    magic = find_content("magic", "h3", soup)
    uses_h2 = find_content("uses", "h2", soup)
    uses_h3 = find_content("uses", "h3", soup)
    
    
    return {
      "success": True,
      "monster_name": monster_name,
      "description": description,
      "abstract": abstract,
      "personality": personality,
      "realm": realm,
      "activities": activities,
      "combat": combat,
      "biology": biology,
      "society": society,
      "abilities": abilities,
      "behavior": behavior,
      "reputation": reputation,
      "culture": culture,
      "sub-races": sub_races,
      "types": types,
      "history": history,
      "rumors": rumors,
      "magic": magic,
      "uses": uses_h2 + uses_h3
      }


if __name__ == "__main__":
  # urls = [
  #   "https://forgottenrealms.fandom.com/wiki/Kenkus",
  # ]
  
  OUTPUT_FILE = "more_data.csv"
  NAME_LIST_PATH = "monster_names.txt"
  ERROR_FILE = "error_more_data.txt"
  
  name_list = load_names(NAME_LIST_PATH)
  
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
      log_error(res, ERROR_FILE)
    else:
      res.pop("success")
      save_success_to_csv(res, OUTPUT_FILE)
        