"""The three systems compared in the paper: keys, labels, source columns, colours."""

# Key used in filenames (docuscope_input/<key>.txt, <key>_tokens.csv).
MODEL_KEYS = ("gpt_4", "gpt_5_2", "rag")

# Key -> label shown in tables, figures and output CSVs.
LABELS = {"gpt_4": "GPT-4o", "gpt_5_2": "GPT-5.2", "rag": "RAG"}
MODELS = tuple(LABELS[key] for key in MODEL_KEYS)

# Key -> column in data/Team6_comparative_analysis.csv.
SOURCE_COLUMNS = {"gpt_4": "A", "gpt_5_2": "Unnamed: 3", "rag": "B"}

# Key -> plot colour, and the same colours by label.
COLOURS = {"gpt_4": "#2a78d6", "gpt_5_2": "#eb6834", "rag": "#1baf7a"}
LABEL_COLOURS = {LABELS[key]: colour for key, colour in COLOURS.items()}
