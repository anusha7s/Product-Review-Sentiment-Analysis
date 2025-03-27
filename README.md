# Sentiment Analysis App

## Overview
This sentiment analysis app utilizes natural language processing (NLP) to analyze product reviews. It provides insights into customer sentiment using AI models, enabling businesses and individuals to understand feedback trends effectively. Users can upload datasets, analyze sentiment in real time, and visualize sentiment distributions through various charts and word clouds.

## Features
- Upload CSV files containing product reviews.
- Sentiment analysis using a transformer-based model (DistilBERT-Base-Uncased-Finetuned-SST-2-English).
- Real-time sentiment prediction for user inputs.
- Visualization tools including word clouds, sentiment distribution charts, and review length analysis.
- Text preprocessing for enhanced accuracy.

## Tech Stack
- **Frontend**: Streamlit for interactive user interface.
- **Backend**: Python for core logic and processing.
- **NLP & Machine Learning**:
  - Hugging Face Transformers for sentiment analysis.
  - TextBlob for polarity and sentiment scoring.
  - WordCloud and NLTK for text visualization and tokenization.
- **Data Handling**: Pandas and NumPy for dataset processing.
- **Visualization**: Matplotlib and Seaborn for data representation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the app:
   ```bash
   streamlit run app.py
   ```
2. Upload a CSV file with a `reviewText` column.
3. Analyze sentiment and explore visualizations.

## Dataset Requirements
- The dataset must be in CSV format.
- It should contain a column named `reviewText` for sentiment analysis.
- Missing values in `reviewText` should be handled before uploading.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

## Future Enhancements
- Multilingual sentiment analysis.
- Fine-tuned domain-specific sentiment models.
- Aspect-based sentiment analysis.
- Real-time interactive sentiment trend dashboard.
- Topic modeling to identify key themes in reviews.
- Recommendation system based on sentiment scores.
- Deployment as a cloud-based service for scalability.

## Contact
For queries or collaborations, reach out via [saraswatanusha99@gmail.com] or open an issue on GitHub.

