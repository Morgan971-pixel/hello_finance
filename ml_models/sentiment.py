from transformers import pipeline

def main():
    # Initialize the sentiment analysis pipeline
    # We specify the finbert model to ensure we get Positive, Negative, and Neutral labels
    print("Loading the sentiment model... (this might take a minute the first time as it downloads)")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    print("\n--- News Headline Sentiment Analyzer ---")
    print("Type 'exit' or 'quit' to stop the program.\n")

    while True:
        # Get the headline from the user
        headline = input("Enter a news headline: ")

        # Check if the user wants to exit
        if headline.lower().strip() in ['exit', 'quit']:
            print("Exiting the analyzer. Goodbye!")
            break

        # Skip empty inputs
        if not headline.strip():
            print("Please enter a valid headline.\n")
            continue

        # Analyze the text
        # The pipeline returns a list of dictionaries, we grab the first one [0]
        result = sentiment_analyzer(headline)[0]

        # Extract and format the label and confidence score
        label = result['label'].capitalize()
        confidence = result['score'] * 100

        # Output the results
        print(f"-> Sentiment: {label} (Confidence: {confidence:.1f}%)\n")

if __name__ == "__main__":
    main()
