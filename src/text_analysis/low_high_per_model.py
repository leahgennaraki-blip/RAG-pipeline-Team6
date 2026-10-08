import pandas as pd
from pathlib import Path

folder = (Path(__file__).resolve().parents[2] / "data" / "qualitative")
absolute_folder = folder.resolve()

# Paste get_highest_lowest_per_model() here

def get_highest_lowest_per_model(absolute_folder):
    """
    Reads the two global mean files and finds the lowest and highest mean value
    for each model.

    Creates:
    1. global_model_criterion_highest_lowest.csv
    - lowest/highest criterion mean per model

    2. global_model_question_criterion_highest_lowest.csv
    - lowest/highest question–criterion mean per model

    All tied lowest/highest values are retained.
    """
    absolute_folder = Path(absolute_folder)
    global_gradings_folder = absolute_folder / "global_gradings"

    model_criterion_path = (
        global_gradings_folder / "global_model_criterion_means.csv"
    )
    model_question_criterion_path = (
        global_gradings_folder / "global_model_question_criterion_means.csv"
    )

    # =========================================================
    # 1) Lowest and highest criterion mean per model
    # =========================================================
    if model_criterion_path.exists():
        df_model_criterion = pd.read_csv(model_criterion_path)

        required_cols = {"Model", "criterion", "mean_value"}
        if not required_cols.issubset(df_model_criterion.columns):
            print(
                f"Skipping {model_criterion_path.name}: "
                f"missing one of {required_cols}"
            )
        else:
            df_model_criterion["mean_value"] = pd.to_numeric(
                df_model_criterion["mean_value"],
                errors="coerce"
            )

            results = []

            for model, group in df_model_criterion.groupby("Model"):
                group = group.dropna(subset=["mean_value"])

                if group.empty:
                    continue

                lowest_value = group["mean_value"].min()
                highest_value = group["mean_value"].max()

                # Keep every criterion tied for lowest value
                lowest_rows = group[group["mean_value"] == lowest_value]
                for _, row in lowest_rows.iterrows():
                    results.append({
                        "Model": model,
                        "value_type": "lowest",
                        "criterion": row["criterion"],
                        "mean_value": row["mean_value"],
                    })

                # Keep every criterion tied for highest value
                highest_rows = group[group["mean_value"] == highest_value]
                for _, row in highest_rows.iterrows():
                    results.append({
                        "Model": model,
                        "value_type": "highest",
                        "criterion": row["criterion"],
                        "mean_value": row["mean_value"],
                    })

            model_criterion_extremes = pd.DataFrame(results).sort_values(
                ["Model", "value_type", "mean_value"]
            )

            output_path = (
                global_gradings_folder
                / "global_model_criterion_highest_lowest.csv"
            )
            model_criterion_extremes.to_csv(output_path, index=False)
            print(f"Saved model-criterion highest/lowest values to: {output_path}")

    else:
        print(f"File not found: {model_criterion_path}")

    # =========================================================
    # 2) Lowest and highest question–criterion mean per model
    # =========================================================
    if model_question_criterion_path.exists():
        df_model_question_criterion = pd.read_csv(model_question_criterion_path)

        required_cols = {"Model", "Question", "criterion", "mean_value"}
        if not required_cols.issubset(df_model_question_criterion.columns):
            print(
                f"Skipping {model_question_criterion_path.name}: "
                f"missing one of {required_cols}"
            )
        else:
            df_model_question_criterion["mean_value"] = pd.to_numeric(
                df_model_question_criterion["mean_value"],
                errors="coerce"
            )

            results = []

            for model, group in df_model_question_criterion.groupby("Model"):
                group = group.dropna(subset=["mean_value"])

                if group.empty:
                    continue

                lowest_value = group["mean_value"].min()
                highest_value = group["mean_value"].max()

                # Keep every Question + criterion combination tied for lowest
                lowest_rows = group[group["mean_value"] == lowest_value]
                for _, row in lowest_rows.iterrows():
                    results.append({
                        "Model": model,
                        "value_type": "lowest",
                        "Question": row["Question"],
                        "criterion": row["criterion"],
                        "mean_value": row["mean_value"],
                    })

                # Keep every Question + criterion combination tied for highest
                highest_rows = group[group["mean_value"] == highest_value]
                for _, row in highest_rows.iterrows():
                    results.append({
                        "Model": model,
                        "value_type": "highest",
                        "Question": row["Question"],
                        "criterion": row["criterion"],
                        "mean_value": row["mean_value"],
                    })

            model_question_criterion_extremes = pd.DataFrame(results).sort_values(
                ["Model", "value_type", "mean_value"]
            )

            output_path = (
                global_gradings_folder
                / "global_model_question_criterion_highest_lowest.csv"
            )
            model_question_criterion_extremes.to_csv(output_path, index=False)
            print(
                "Saved model-question-criterion highest/lowest values to: "
                f"{output_path}"
            )

    else:
        print(f"File not found: {model_question_criterion_path}")


# Run the function
get_highest_lowest_per_model(absolute_folder)



def get_lowest_mean_per_model_per_criterion(absolute_folder):
    """
    For every Model and criterion:
    - find the lowest mean_value across questions
    - retain the Question where it occurs
    - retain all tied lowest values

    Input:
    global_gradings/global_model_question_criterion_means.csv

    Output:
    global_gradings/lowest_mean_per_model_per_criterion.csv
    """
    absolute_folder = Path(absolute_folder)

    input_path = (
        absolute_folder
        / "global_gradings"
        / "global_model_question_criterion_means.csv"
    )

    output_path = (
        absolute_folder
        / "global_gradings"
        / "lowest_mean_per_model_per_criterion.csv"
    )

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    required_cols = {"Model", "Question", "criterion", "mean_value"}
    if not required_cols.issubset(df.columns):
        print(f"Input file must contain these columns: {required_cols}")
        print(f"Found: {list(df.columns)}")
        return

    # Ensure scores are numeric and remove missing scores
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df = df.dropna(subset=["mean_value"])

    # Find the minimum mean value within every Model + criterion group
    lowest_values = (
        df.groupby(["Model", "criterion"])["mean_value"]
        .transform("min")
    )

    # Retain rows equal to their group's minimum.
    # This also retains all ties.
    lowest_df = df[df["mean_value"] == lowest_values].copy()

    # Clear ordering for the output file
    lowest_df = lowest_df.sort_values(
        by=["Model", "criterion", "mean_value", "Question"]
    )

    lowest_df.to_csv(output_path, index=False)

    print(f"Saved lowest means per model and criterion to:\n{output_path}")
    print("\nPreview:")
    print(lowest_df.to_string(index=False))

    return lowest_df


get_lowest_mean_per_model_per_criterion(absolute_folder)


def get_worst_criterion_per_model(absolute_folder):
    """
    Finds the criterion with the lowest overall mean score for each model.

    Input:
    global_gradings/global_model_criterion_means.csv

    Output:
    global_gradings/worst_criterion_per_model.csv

    Ties are retained: if a model has multiple criteria with the same
    lowest mean, all of them will be saved.
    """
    absolute_folder = Path(absolute_folder)

    input_path = (
        absolute_folder
        / "global_gradings"
        / "global_model_criterion_means.csv"
    )

    output_path = (
        absolute_folder
        / "global_gradings"
        / "worst_criterion_per_model.csv"
    )

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    required_cols = {"Model", "criterion", "mean_value"}
    if not required_cols.issubset(df.columns):
        print(f"Input file must contain: {required_cols}")
        print(f"Columns found: {list(df.columns)}")
        return

    # Ensure the mean values are numeric
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df = df.dropna(subset=["mean_value"])

    # Find the lowest mean_value within each Model
    lowest_per_model = df.groupby("Model")["mean_value"].transform("min")

    # Keep the criterion/criteria with that lowest value.
    # This retains tied criteria if applicable.
    worst_criteria = df[df["mean_value"] == lowest_per_model].copy()

    worst_criteria = worst_criteria.sort_values(
        by=["Model", "mean_value", "criterion"]
    )

    worst_criteria.to_csv(output_path, index=False)

    print(f"Saved worst-performing criterion per model to:\n{output_path}")
    print("\nResults:")
    print(worst_criteria.to_string(index=False))

    return worst_criteria


get_worst_criterion_per_model(absolute_folder)
