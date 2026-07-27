import pandas as pd
from pathlib import Path

folder = Path("qualitative_starter_final_gradings")  # relative path
absolute_folder = folder.resolve()

print("Current working directory:", Path.cwd())
print("Absolute folder path:", absolute_folder)

def load_csv_files_from_folder(absolute_folder):
    csv_files = list(absolute_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {absolute_folder}.")
        return []
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded {file.name} successfully.")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    return dataframes


load_csv_files_from_folder(absolute_folder)


def add_id_from_first_two_columns(df, id_col_name="id", sep=" "):
    
    cols = df.columns
    if len(cols) < 2:
        raise ValueError("DataFrame must have at least two columns to build an id.")
    
    first, second = cols[0], cols[1]
    df[id_col_name] = df[first].astype(str) + sep + df[second].astype(str)
    return df

def drop_rows_with_no_values_after_second_column(df, id_col_name="id", empty_threshold=8):
    """
    Drop rows where `empty_threshold` grading columns (after the first two) are
    NaN or '-'. By default, empty_threshold=8.

    Assumes:
    - First two columns are identifiers (e.g. Question, Model).
    - `id_col_name` is the name of the id column added later.
    """
    cols = df.columns.tolist()
    if len(cols) <= 2:
        # Nothing to check beyond the first two columns
        return df
    
    # Grading columns: all columns except first two and the id column
    graded_cols = [c for c in cols if c not in (cols[0], cols[1], id_col_name)]
    if not graded_cols:
        return df
    
    # Boolean mask: True where value is NaN or "-"
    empty_or_dash = df[graded_cols].isna() | (df[graded_cols] == "-")
    
    # Count how many grading columns are empty/'-' in each row
    empty_count = empty_or_dash.sum(axis=1)
    
    # Keep rows where fewer than `empty_threshold` grading columns are empty
    df_clean = df[empty_count < empty_threshold].copy()
    return df_clean


def create_clean_csv_files(absolute_folder):
    
    cleaned_folder = absolute_folder / "cleaned"
    cleaned_folder.mkdir(exist_ok=True)
    
    csv_files = list(absolute_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {absolute_folder}.")
        return
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Remember original first two column names (Question, Model)
            first_col, second_col = df.columns[0], df.columns[1]
            
            # Add id and drop rows based on grading columns
            df = add_id_from_first_two_columns(df, id_col_name="id")
            df = drop_rows_with_no_values_after_second_column(df, id_col_name="id", empty_threshold=8)
            
            
            # >>> THIS PART MAKES 'id' THE FIRST COLUMN <<<
            cols = df.columns.tolist()
            cols = ["id"] + [c for c in cols if c != "id"]
            df = df[cols]
            # ---------------------------------------------
            
            # Save with '_cleaned' appended
            output_name = f"{file.stem}_cleaned{file.suffix}"
            output_path = cleaned_folder / output_name
            df.to_csv(output_path, index=False)
            print(f"Saved cleaned file to {output_path}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")


# Run the cleaning steps for CSV files.

dfs = load_csv_files_from_folder(absolute_folder)

dfs_processed = []
for df in dfs:
    df = add_id_from_first_two_columns(df)
    df = drop_rows_with_no_values_after_second_column(df)
    dfs_processed.append(df)

for i, df in enumerate(dfs_processed):
    print(f"\nCleaned DataFrame {i} head:")
    print(df.head())

create_clean_csv_files(absolute_folder)

def compute_stats_for_cleaned_files(absolute_folder):
    cleaned_folder = absolute_folder / "cleaned"
    if not cleaned_folder.exists():
        print(f"No 'cleaned' folder found at {cleaned_folder}. Run create_clean_csv_files() first.")
        return
    
    csv_files = list(cleaned_folder.glob("*.csv"))
    if not csv_files:
        print(f"No cleaned CSV files found in {cleaned_folder}.")
        return
    
    for file in csv_files:
        print("\n" + "=" * 80)
        print(f"File: {file.name}")
        print("=" * 80)
        
        # Read cleaned file, treat "-" as NaN
        df = pd.read_csv(file, na_values=["-"])
        
        # Skip files without id
        if "id" not in df.columns:
            print(f"Skipping {file.name}: no 'id' column. Columns: {list(df.columns)}")
            continue
        
        # Identify non-grading columns
        non_grade_cols = ["id"]
        for c in ["Question", "Model"]:
            if c in df.columns:
                non_grade_cols.append(c)
        
        # Grading columns are everything except id / Question / Model
        grade_cols = [c for c in df.columns if c not in non_grade_cols]
        
        # Ensure grading columns are numeric
        df[grade_cols] = df[grade_cols].apply(pd.to_numeric, errors="coerce")
        
        # ---- Row-wise stats (use all rows) ----
        row_means    = []
        row_mins     = []
        row_maxes    = []
        row_min_cols = []
        row_max_cols = []
        
        for idx, row in df[grade_cols].iterrows():
            s = row.dropna()
            if len(s) == 0:
                # No valid grades for this row
                row_means.append(pd.NA)
                row_mins.append(pd.NA)
                row_maxes.append(pd.NA)
                row_min_cols.append(pd.NA)
                row_max_cols.append(pd.NA)
            else:
                row_means.append(s.mean())
                row_mins.append(s.min())
                row_maxes.append(s.max())
                row_min_cols.append(s.idxmin())  # column name of min
                row_max_cols.append(s.idxmax())  # column name of max
        
        row_stats = pd.DataFrame({
            "id": df["id"],
            "row_mean": pd.Series(row_means).round(3),
            "row_min": pd.Series(row_mins).round(3),
            "row_max": pd.Series(row_maxes).round(3),
            "row_min_col": row_min_cols,
            "row_max_col": row_max_cols,
        })
        
        print("\nRow-wise statistics (per id):")
        print(row_stats.to_string(index=False))
        
        # ---- Column-wise stats (EXCLUDE rows where Model == "RAG") ----
        if "Model" in df.columns:
            rag_mask = df["Model"] == "RAG"
        else:
            rag_mask = pd.Series(False, index=df.index)
        
        df_no_rag = df.loc[~rag_mask, grade_cols]
        
        col_means = []
        col_mins  = []
        col_maxes = []
        min_ids   = []
        max_ids   = []
        
        for col in grade_cols:
            s = df_no_rag[col]
            if s.notna().any():
                # Stats for this grading column
                col_means.append(s.mean(skipna=True))
                col_mins.append(s.min(skipna=True))
                col_maxes.append(s.max(skipna=True))
                
                idx_min = s.idxmin()
                idx_max = s.idxmax()
                
                min_ids.append(df.loc[idx_min, "id"])
                max_ids.append(df.loc[idx_max, "id"])
            else:
                col_means.append(pd.NA)
                col_mins.append(pd.NA)
                col_maxes.append(pd.NA)
                min_ids.append(pd.NA)
                max_ids.append(pd.NA)
        
        col_stats = pd.DataFrame({
            "column": grade_cols,
            "col_mean": pd.Series(col_means, dtype="Float64").round(3),
            "col_min": pd.Series(col_mins, dtype="Float64").round(3),
            "col_max": pd.Series(col_maxes, dtype="Float64").round(3),
            "min_id": min_ids,
            "max_id": max_ids,
})
        
        print("\nColumn-wise statistics (per grading column, excluding Model == 'RAG'):")
        print(col_stats.to_string(index=False))

# Run stats computation on the cleaned files
compute_stats_for_cleaned_files(absolute_folder)

def compute_model_criteria_stats(absolute_folder):
    """
    For each cleaned CSV:
    - For each Model
    - For each grading column (header)
    
    Compute:
    - mean, min, max values
    - Question where min occurs
    - Question where max occurs
    
    Prints a summary table per file.
    """
    cleaned_folder = absolute_folder / "cleaned"
    if not cleaned_folder.exists():
        print(f"No 'cleaned' folder found at {cleaned_folder}. Run create_clean_csv_files() first.")
        return
    
    csv_files = list(cleaned_folder.glob("*.csv"))
    if not csv_files:
        print(f"No cleaned CSV files found in {cleaned_folder}.")
        return
    
    for file in csv_files:
        print("\n" + "=" * 80)
        print(f"Model-wise criteria stats for file: {file.name}")
        print("=" * 80)
        
        # Read cleaned file, treat "-" as NaN
        df = pd.read_csv(file, na_values=["-"])
        
        # Basic checks
        if "Model" not in df.columns or "Question" not in df.columns:
            print("File missing 'Model' or 'Question' column; skipping.")
            continue
        
        # Identify grading columns (exclude id, Question, Model)
        non_grade_cols = ["id", "Question", "Model"]
        grade_cols = [c for c in df.columns if c not in non_grade_cols]
        
        # Ensure grading columns are numeric
        df[grade_cols] = df[grade_cols].apply(pd.to_numeric, errors="coerce")
        
        results = []
        
        # Loop over models
        for model in df["Model"].unique():
            df_model = df[df["Model"] == model]
            if df_model.empty:
                continue
            
            for col in grade_cols:
                s = df_model[col]
                # Skip if no valid values for this model & header
                if not s.notna().any():
                    continue
                
                mean_val = s.mean(skipna=True)
                min_val  = s.min(skipna=True)
                max_val  = s.max(skipna=True)
                
                # idxmin / idxmax give index in df_model; use that to get Question
                idx_min = s.idxmin()
                idx_max = s.idxmax()
                
                min_question = df_model.loc[idx_min, "Question"]
                max_question = df_model.loc[idx_max, "Question"]
                
                results.append({
                    "model": model,
                    "header": col,
                    "mean": round(mean_val, 3),
                    "min": round(min_val, 3),
                    "max": round(max_val, 3),
                    "min_question": min_question,
                    "max_question": max_question,
                })
        
        if results:
            stats_df = pd.DataFrame(results)
            # Optional: sort for readability
            stats_df = stats_df.sort_values(by=["model", "header"])
            print(stats_df.to_string(index=False))
        else:
            print("No grading data found for any model in this file.")


compute_model_criteria_stats(absolute_folder)

def save_summary_stats_per_file(absolute_folder):
    """
    For each cleaned CSV in absolute_folder / 'cleaned':

      1) Row-wise stats (per id)  -> {stem}_row_stats.csv
      2) Column-wise stats        -> {stem}_column_stats.csv
      3) Model-wise criteria stats-> {stem}_model_criteria_stats.csv

    All files are written into absolute_folder / 'summary_stats'.
    """

    absolute_folder = Path(absolute_folder)
    cleaned_folder = absolute_folder / "cleaned"
    if not cleaned_folder.exists():
        print(f"No 'cleaned' folder found at {cleaned_folder}. Run create_clean_csv_files() first.")
        return

    csv_files = list(cleaned_folder.glob("*.csv"))
    if not csv_files:
        print(f"No cleaned CSV files found in {cleaned_folder}.")
        return

    summary_folder = absolute_folder / "summary_stats"
    summary_folder.mkdir(exist_ok=True)

    for file in csv_files:
        print(f"Processing summary stats for: {file.name}")
        df = pd.read_csv(file, na_values=["-"])

        # --- identify grading columns ---
        if "id" not in df.columns:
            print(f"  Skipping {file.name}: no 'id' column.")
            continue

        non_grade_cols = ["id"]
        for c in ["Question", "Model"]:
            if c in df.columns:
                non_grade_cols.append(c)
        grade_cols = [c for c in df.columns if c not in non_grade_cols]

        if not grade_cols:
            print(f"  Skipping {file.name}: no grading columns found.")
            continue

        # Ensure grading columns are numeric
        df[grade_cols] = df[grade_cols].apply(pd.to_numeric, errors="coerce")

        stem = file.stem

        # ======================================================
        # 1) ROW-WISE STATS (per id)
        # ======================================================
        row_means    = []
        row_mins     = []
        row_maxes    = []
        row_min_cols = []
        row_max_cols = []

        for _, row in df[grade_cols].iterrows():
            s = row.dropna()
            if len(s) == 0:
                row_means.append(pd.NA)
                row_mins.append(pd.NA)
                row_maxes.append(pd.NA)
                row_min_cols.append(pd.NA)
                row_max_cols.append(pd.NA)
            else:
                row_means.append(s.mean())
                row_mins.append(s.min())
                row_maxes.append(s.max())
                row_min_cols.append(s.idxmin())
                row_max_cols.append(s.idxmax())

        row_stats = pd.DataFrame({
            "id": df["id"],
            "row_mean": pd.Series(row_means).round(3),
            "row_min": pd.Series(row_mins).round(3),
            "row_max": pd.Series(row_maxes).round(3),
            "row_min_col": row_min_cols,
            "row_max_col": row_max_cols,
        })

        row_out = summary_folder / f"{stem}_row_stats.csv"
        row_stats.to_csv(row_out, index=False)
        print(f"  Saved row-wise stats to: {row_out}")

        # ======================================================
        # 2) COLUMN-WISE STATS (exclude Model == 'RAG')
        # ======================================================
        if "Model" in df.columns:
            rag_mask = df["Model"] == "RAG"
        else:
            rag_mask = pd.Series(False, index=df.index)

        df_no_rag = df.loc[~rag_mask, grade_cols]

        col_means = []
        col_mins  = []
        col_maxes = []
        min_ids   = []
        max_ids   = []

        for col in grade_cols:
            s = df_no_rag[col]
            if s.notna().any():
                col_means.append(s.mean(skipna=True))
                col_mins.append(s.min(skipna=True))
                col_maxes.append(s.max(skipna=True))

                idx_min = s.idxmin()
                idx_max = s.idxmax()

                min_ids.append(df.loc[idx_min, "id"])
                max_ids.append(df.loc[idx_max, "id"])
            else:
                col_means.append(pd.NA)
                col_mins.append(pd.NA)
                col_maxes.append(pd.NA)
                min_ids.append(pd.NA)
                max_ids.append(pd.NA)

        col_stats = pd.DataFrame({
            "column": grade_cols,
            "col_mean": pd.Series(col_means, dtype="Float64").round(3),
            "col_min": pd.Series(col_mins, dtype="Float64").round(3),
            "col_max": pd.Series(col_maxes, dtype="Float64").round(3),
            "min_id": min_ids,
            "max_id": max_ids,
        })

        col_out = summary_folder / f"{stem}_column_stats.csv"
        col_stats.to_csv(col_out, index=False)
        print(f"  Saved column-wise stats to: {col_out}")

        # ======================================================
        # 3) MODEL-WISE CRITERIA STATS (per Model x criterion)
        # ======================================================
        if ("Model" in df.columns) and ("Question" in df.columns):
            results = []

            for model in df["Model"].unique():
                df_model = df[df["Model"] == model]
                if df_model.empty:
                    continue

                for col in grade_cols:
                    s = df_model[col]
                    if not s.notna().any():
                        continue

                    mean_val = s.mean(skipna=True)
                    min_val  = s.min(skipna=True)
                    max_val  = s.max(skipna=True)

                    idx_min = s.idxmin()
                    idx_max = s.idxmax()

                    min_question = df_model.loc[idx_min, "Question"]
                    max_question = df_model.loc[idx_max, "Question"]

                    results.append({
                        "model": model,
                        "criterion": col,
                        "mean": round(mean_val, 3),
                        "min": round(min_val, 3),
                        "max": round(max_val, 3),
                        "min_question": min_question,
                        "max_question": max_question,
                    })

            if results:
                model_crit_stats = pd.DataFrame(results)
                model_out = summary_folder / f"{stem}_model_criteria_stats.csv"
                model_crit_stats.to_csv(model_out, index=False)
                print(f"  Saved model-wise criteria stats to: {model_out}")
            else:
                print("  No grading data found for any model in this file (for model-wise stats).")

    print("Done saving per-file summary statistics.")




def compute_global_means_separate_files(
    absolute_folder,
    model_criterion_filename="global_model_criterion_means.csv",
    model_question_criterion_filename="global_model_question_criterion_means.csv",
):
    """
    From all cleaned CSV files in absolute_folder / 'cleaned':
    1) Compute mean for every model per criterion (header).
    2) Compute mean for every criterion per question per model.
    
    Save results into two separate CSV files:
    - model_criterion_filename
    - model_question_criterion_filename
    """
    cleaned_folder = absolute_folder / "cleaned"
    if not cleaned_folder.exists():
        print(f"No 'cleaned' folder found at {cleaned_folder}. Run create_clean_csv_files() first.")
        return
    
    csv_files = list(cleaned_folder.glob("*.csv"))
    if not csv_files:
        print(f"No cleaned CSV files found in {cleaned_folder}.")
        return
    
    long_dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file, na_values=["-"])
        
        # Check required columns
        if "Model" not in df.columns or "Question" not in df.columns:
            print(f"Skipping {file.name}: missing 'Model' or 'Question' column.")
            continue
        
        # Identify grading (criterion) columns
        non_grade_cols = ["id", "Question", "Model"]
        grade_cols = [c for c in df.columns if c not in non_grade_cols]
        
        if not grade_cols:
            print(f"Skipping {file.name}: no grading columns found.")
            continue
        
        # Ensure grading columns are numeric
        df[grade_cols] = df[grade_cols].apply(pd.to_numeric, errors="coerce")
        
        # Melt to long format: one row per (Question, Model, criterion, value)
        df_long = df.melt(
            id_vars=["Question", "Model"],
            value_vars=grade_cols,
            var_name="criterion",
            value_name="value",
        )
        df_long["source_file"] = file.name
        
        long_dfs.append(df_long)
    
    if not long_dfs:
        print("No usable data found in cleaned CSVs.")
        return
    
    all_long = pd.concat(long_dfs, ignore_index=True)
    
    # Keep only rows with numeric values
    all_long = all_long.dropna(subset=["value"])

    global_gradings_folder = absolute_folder / "global_gradings"
    global_gradings_folder.mkdir(exist_ok=True)
    
    # ---------------------------------------------------
    # 1) Mean for every model per criterion (all files combined)
    # ---------------------------------------------------

    model_criterion = (
        all_long
        .groupby(["Model", "criterion"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_value"})
    )
    
    model_criterion_path = global_gradings_folder / model_criterion_filename
    model_criterion.to_csv(model_criterion_path, index=False)
    print(f"Saved model-per-criterion means to {model_criterion_path}")
    
    # ---------------------------------------------------
    # 2) Mean for every criterion per question per model (all files combined)
    # ---------------------------------------------------

    model_question_criterion = (
        all_long
        .groupby(["Model", "Question", "criterion"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_value"})
    )
    
    model_question_criterion_path = global_gradings_folder / model_question_criterion_filename
    model_question_criterion.to_csv(model_question_criterion_path, index=False)
    print(f"Saved criterion-per-question-per-model means to {model_question_criterion_path}")
    
    return model_criterion, model_question_criterion

compute_global_means_separate_files(absolute_folder)

save_summary_stats_per_file(absolute_folder)