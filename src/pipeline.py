import csv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from collections import Counter
import spacy
from pathlib import Path
from typing import List, Any

nlp = spacy.load("en_core_web_sm")

class MonsterLoader:
    def __init__(self, csv_path: str):
      self.csv_path = csv_path
      
    def load(self) -> List[Any]:
      monsters = []
      with open(self.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
          monsters.append({"name": row["monster_name"], "description": row["description"]})
      return monsters

class EmbeddingService:
  def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
    self.model= SentenceTransformer(model_name)
    
  def embed(self, texts):
    return self.model.encode(texts, convert_to_numpy=True)

class VectorIndex:
  def __init__(self, dim: int):
    self.index = faiss.IndexFlatL2(dim)
  

  def add(self, embeddings):
    embeddings = self._ensure_2d(embeddings).astype("float32")
    self.index.add(embeddings)

  def search(self, vector, k=5):
    vector = self._ensure_2d(vector).astype("float32")
    dist, idx = self.index.search(vector, k)
    return idx[0]

  def _ensure_2d(self, arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim == 1:
      return arr.reshape(1, -1)
    return arr

class TermExpander:
  def __init__(self):
    self.bad_terms = []
   
  def extract(self, text: str, top_k=20):
    doc = nlp(text)
    bag = Counter()

    for token in doc:
      if token.pos_ not in ("NOUN", "PROPN"):
        continue
      if len(token.lemma_) < 3:
        continue

      if token.is_digit or token.is_punct:
        continue

      lemma = token.lemma_.lower()
      if lemma in self.bad_terms:
        continue
      
      if not lemma.isalpha():
        continue
      bag.update([lemma])

    return [t for t, _ in bag.most_common(top_k)]

def expand_monster_terms(csv_path: str, monster_name: str, top_k=20) -> List[str]:
    loader = MonsterLoader(csv_path)
    monsters = loader.load()
    target = next((monster for monster in monsters if monster["name"].lower() == monster_name.lower()), None)

    if not target:
      return [monster_name]

    description = target["description"]
    expander = TermExpander()
    terms = expander.extract(description, top_k=top_k)
    return terms
  
def expand_monster(monster_name: str):
  return expand_monster_terms("monster_descriptions.csv", monster_name, 10)

def format_monster(monster_name):
  termos = expand_monster(monster_name)
  if not termos:
    return monster_name
  termos_txt = ", ".join(termos)
  return f"{monster_name} - Description of {monster_name}: ({termos_txt})"

#Só para testes
if __name__ == "__main__":
    monster_name = "Aarakocra"
    csv_path = "monster_descriptions.csv"

    #Recupera os termos relavantes presentes nas descrições
    terms = expand_monster_terms(csv_path, monster_name, top_k=10)
  
    print(f"\nExpanded '{monster_name}':")
    print(terms)