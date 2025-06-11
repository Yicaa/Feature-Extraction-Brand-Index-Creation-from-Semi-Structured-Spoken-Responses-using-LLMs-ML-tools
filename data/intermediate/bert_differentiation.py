# bert_differentiation.py

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# 1. Load the input CSV from R
df = pd.read_csv("data/intermediate/tidy_join_transcript2.csv")  # adapt path if needed
texts = df["main_audio"].astype(str).tolist()

# 2. Define general-purpose differentiation prompts (in Spanish)
prompts = [
    "Este banco es diferente a los demás.",
    "Tiene algo único que lo distingue del resto.",
    "Se nota que no es como otros bancos.",
    "Ofrece una propuesta distinta a lo habitual.",
    "No es un banco convencional.",
    "Tiene una manera propia de hacer las cosas.",
    "Es claramente un banco con enfoque diferente."
]

# 3. Load the multilingual BERT model
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# 4. Encode texts and prompts
text_embeddings = model.encode(texts, convert_to_tensor=True)
prompt_embeddings = model.encode(prompts, convert_to_tensor=True)

# 5. Compute cosine similarity
similarity_matrix = util.cos_sim(text_embeddings, prompt_embeddings).cpu().numpy()

# 6. Take the maximum similarity score for each respondent
max_similarities = np.max(similarity_matrix, axis=1)

# 7. Rescale to 1–7 differentiation index
scaled_scores = 1 + 6 * (max_similarities - np.min(max_similarities)) / (np.max(max_similarities) - np.min(max_similarities))
df["differentiation_index"] = np.round(scaled_scores)

# 8. Save output for R
df[["respondent_id", "differentiation_index"]].to_csv("data/intermediate/differentiation_index.csv", index=False)

print("✅ Differentiation index saved to data/intermediate/differentiation_index.csv")
