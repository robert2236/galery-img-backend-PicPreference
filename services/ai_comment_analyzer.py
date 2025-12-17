from transformers import pipeline
from collections import Counter
import re

class CommentAIAnalyzer:
    def __init__(self):
        
        # Analizador de sentimiento
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Extractor de palabras clave (simplificado)
        self.stop_words = {"el", "la", "los", "las", "un", "una", "de", "en", "y", "que"}
    
    def analyze_comments(self, comments_list):
        """Analiza una lista de comentarios de una imagen"""
        all_comments_text = " ".join([c['text'] for c in comments_list])
        
        if not all_comments_text.strip():
            return {
                "sentiment_score": 0.0,
                "keywords": [],
                "popularity_score": 0.0
            }
        
        # 1. Análisis de sentimiento promedio
        sentiments = []
        for comment in comments_list:
            result = self.sentiment_analyzer(comment['text'][:512])[0]
            # Convertir a escala -1 a 1
            if result['label'] == '1 star':
                sentiments.append(-1.0)
            elif result['label'] == '2 stars':
                sentiments.append(-0.5)
            elif result['label'] == '3 stars':
                sentiments.append(0.0)
            elif result['label'] == '4 stars':
                sentiments.append(0.5)
            else:
                sentiments.append(1.0)
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # 2. Extraer palabras clave
        words = re.findall(r'\b\w+\b', all_comments_text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        word_freq = Counter(words)
        keywords = [word for word, count in word_freq.most_common(5)]
        
        # 3. Calcular puntaje de popularidad
        num_comments = len(comments_list)
        popularity = min(num_comments / 10, 1.0)  # Normalizado a 0-1
        
        return {
            "sentiment_score": float(avg_sentiment),
            "keywords": keywords,
            "popularity_score": float(popularity)
        }