def main() -> None:
    # Imported lazily so that importing the package does not load NLTK
    from un_general_debate_analysis.preprocessing import main as run_preprocessing

    run_preprocessing()
