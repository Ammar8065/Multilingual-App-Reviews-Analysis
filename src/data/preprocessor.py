"""
Text preprocessing components for multilingual text cleaning and normalization.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import pandas as pd
from src.data.interfaces import TextPreprocessorInterface
from src.config import get_config
from src.utils.logger import get_logger


class TextPreprocessor(TextPreprocessorInterface):
    """Multilingual text preprocessor for cleaning and normalization."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Language-specific patterns and rules
        self.language_patterns = {
            'en': {
                'contractions': {
                    "won't": "will not", "can't": "cannot", "n't": " not",
                    "'re": " are", "'ve": " have", "'ll": " will",
                    "'d": " would", "'m": " am", "it's": "it is",
                    "that's": "that is", "what's": "what is"
                }
            },
            'es': {
                'accents': True,
                'special_chars': ['ñ', 'Ñ', 'ü', 'Ü']
            },
            'fr': {
                'accents': True,
                'special_chars': ['ç', 'Ç', 'œ', 'Œ', 'æ', 'Æ']
            },
            'de': {
                'special_chars': ['ß', 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü']
            }
        }
        
        # Common patterns for all languages
        self.common_patterns = {
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_numbers': re.compile(r'[\+]?[1-9]?[0-9]{7,15}'),
            'mentions': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtags': re.compile(r'#[A-Za-z0-9_]+'),
            'extra_whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\-.,!?;:()\[\]{}"\']'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'numbers': re.compile(r'\b\d+\b')
        }
    
    def normalize_encoding(self, text: str) -> str:
        """
        Normalize text encoding to handle various character encodings.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Normalize Unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Handle common encoding issues
            encoding_fixes = {
                'â€™': "'",  # Smart apostrophe
                'â€œ': '"',  # Smart quote left
                'â€': '"',   # Smart quote right
                'â€"': '—',  # Em dash
                'â€"': '–',  # En dash
                'Ã¡': 'á',   # á with encoding issue
                'Ã©': 'é',   # é with encoding issue
                'Ã­': 'í',   # í with encoding issue
                'Ã³': 'ó',   # ó with encoding issue
                'Ãº': 'ú',   # ú with encoding issue
                'Ã±': 'ñ',   # ñ with encoding issue
            }
            
            for wrong, correct in encoding_fixes.items():
                text = text.replace(wrong, correct)
            
            # Remove or replace non-printable characters
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"Error normalizing encoding: {str(e)}")
            return str(text).strip()
    
    def clean_text(self, text: str, language: str = 'en') -> str:
        """
        Clean and normalize text for the specified language.
        
        Args:
            text: Input text to clean
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned text string
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and normalize encoding
        text = self.normalize_encoding(str(text))
        
        # Basic cleaning steps
        text = self._basic_cleaning(text)
        
        # Language-specific cleaning
        text = self._language_specific_cleaning(text, language)
        
        # Final normalization
        text = self._final_normalization(text)
        
        return text
    
    def _basic_cleaning(self, text: str) -> str:
        """Apply basic cleaning steps common to all languages."""
        
        # Remove URLs
        text = self.common_patterns['urls'].sub(' [URL] ', text)
        
        # Remove email addresses
        text = self.common_patterns['emails'].sub(' [EMAIL] ', text)
        
        # Remove phone numbers
        text = self.common_patterns['phone_numbers'].sub(' [PHONE] ', text)
        
        # Handle social media mentions and hashtags
        text = self.common_patterns['mentions'].sub(' [MENTION] ', text)
        text = self.common_patterns['hashtags'].sub(' [HASHTAG] ', text)
        
        # Fix repeated characters (e.g., "sooooo" -> "so")
        text = self.common_patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def _language_specific_cleaning(self, text: str, language: str) -> str:
        """Apply language-specific cleaning rules."""
        
        lang_rules = self.language_patterns.get(language, {})
        
        # Handle contractions (mainly for English)
        if language == 'en' and 'contractions' in lang_rules:
            for contraction, expansion in lang_rules['contractions'].items():
                text = text.replace(contraction, expansion)
        
        # Preserve language-specific characters
        if 'special_chars' in lang_rules:
            # This is handled in the normalization step
            pass
        
        # Language-specific patterns
        if language == 'zh':  # Chinese
            # Preserve Chinese characters and punctuation
            text = re.sub(r'[^\u4e00-\u9fff\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'ja':  # Japanese
            # Preserve Japanese characters
            text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'ko':  # Korean
            # Preserve Korean characters
            text = re.sub(r'[^\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'ar':  # Arabic
            # Preserve Arabic characters
            text = re.sub(r'[^\u0600-\u06ff\u0750-\u077f\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'hi':  # Hindi
            # Preserve Devanagari script
            text = re.sub(r'[^\u0900-\u097f\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'th':  # Thai
            # Preserve Thai characters
            text = re.sub(r'[^\u0e00-\u0e7f\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        elif language == 'ru':  # Russian
            # Preserve Cyrillic characters
            text = re.sub(r'[^\u0400-\u04ff\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
        else:
            # For Latin-based languages, remove special characters but preserve accents
            text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'\u00c0-\u017f]', ' ', text)
        
        return text
    
    def _final_normalization(self, text: str) -> str:
        """Apply final normalization steps."""
        
        # Normalize whitespace
        text = self.common_patterns['extra_whitespace'].sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Handle empty or very short texts
        if len(text) < self.config.data.min_text_length:
            return ""
        
        # Truncate very long texts
        if len(text) > self.config.data.max_text_length:
            text = text[:self.config.data.max_text_length].rsplit(' ', 1)[0]
        
        return text
    
    def tokenize_multilingual(self, text: str, language: str = 'en') -> List[str]:
        """
        Tokenize text for the specified language.
        
        Args:
            text: Input text to tokenize
            language: Language code for language-specific tokenization
            
        Returns:
            List of tokens
        """
        if not text or pd.isna(text):
            return []
        
        text = str(text).strip()
        
        # Basic tokenization for different language families
        if language in ['zh', 'ja']:
            # For Chinese and Japanese, character-level tokenization might be needed
            # For now, use simple whitespace tokenization
            tokens = self._cjk_tokenization(text, language)
        elif language == 'th':
            # Thai doesn't use spaces between words
            tokens = self._thai_tokenization(text)
        else:
            # Standard whitespace and punctuation tokenization
            tokens = self._standard_tokenization(text)
        
        # Filter out empty tokens and very short tokens
        tokens = [token for token in tokens if len(token.strip()) > 1]
        
        return tokens
    
    def _standard_tokenization(self, text: str) -> List[str]:
        """Standard tokenization for most languages."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _cjk_tokenization(self, text: str, language: str) -> List[str]:
        """Tokenization for Chinese, Japanese, Korean languages."""
        # Simple approach - split on whitespace and punctuation
        # In production, you'd want to use language-specific tokenizers
        tokens = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+|\w+', text)
        return tokens
    
    def _thai_tokenization(self, text: str) -> List[str]:
        """Tokenization for Thai language."""
        # Simple approach - in production, use pythainlp or similar
        tokens = re.findall(r'[\u0e00-\u0e7f]+|\w+', text)
        return tokens
    
    def batch_clean_text(self, texts: List[str], languages: List[str]) -> List[str]:
        """
        Clean multiple texts in batch.
        
        Args:
            texts: List of texts to clean
            languages: List of corresponding language codes
            
        Returns:
            List of cleaned texts
        """
        if len(texts) != len(languages):
            self.logger.warning("Texts and languages lists have different lengths")
            languages = languages * (len(texts) // len(languages) + 1)
            languages = languages[:len(texts)]
        
        cleaned_texts = []
        for text, language in zip(texts, languages):
            try:
                cleaned_text = self.clean_text(text, language)
                cleaned_texts.append(cleaned_text)
            except Exception as e:
                self.logger.warning(f"Error cleaning text: {str(e)}")
                cleaned_texts.append("")
        
        return cleaned_texts
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text', 
                           language_column: str = 'review_language') -> pd.DataFrame:
        """
        Preprocess text data in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to preprocess
            language_column: Name of the language column
            
        Returns:
            DataFrame with preprocessed text
        """
        df_processed = df.copy()
        
        if text_column not in df.columns:
            self.logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df_processed
        
        if language_column not in df.columns:
            self.logger.warning(f"Language column '{language_column}' not found, using 'en' as default")
            languages = ['en'] * len(df)
        else:
            languages = df[language_column].fillna('en').tolist()
        
        self.logger.info(f"Preprocessing {len(df)} texts...")
        
        # Clean texts
        texts = df[text_column].fillna('').tolist()
        cleaned_texts = self.batch_clean_text(texts, languages)
        
        # Add cleaned text column
        df_processed[f'{text_column}_cleaned'] = cleaned_texts
        
        # Add text statistics
        df_processed[f'{text_column}_length'] = [len(text) for text in cleaned_texts]
        df_processed[f'{text_column}_word_count'] = [
            len(text.split()) if text else 0 for text in cleaned_texts
        ]
        
        # Filter out empty texts if requested
        empty_texts = df_processed[f'{text_column}_cleaned'] == ""
        if empty_texts.sum() > 0:
            self.logger.warning(f"Found {empty_texts.sum()} empty texts after cleaning")
        
        self.logger.info("Text preprocessing completed")
        
        return df_processed
    
    def get_preprocessing_stats(self, original_texts: List[str], 
                              cleaned_texts: List[str]) -> Dict[str, any]:
        """
        Generate preprocessing statistics.
        
        Args:
            original_texts: List of original texts
            cleaned_texts: List of cleaned texts
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'total_texts': len(original_texts),
            'empty_after_cleaning': sum(1 for text in cleaned_texts if not text),
            'avg_length_before': sum(len(str(text)) for text in original_texts) / len(original_texts),
            'avg_length_after': sum(len(text) for text in cleaned_texts) / len(cleaned_texts),
            'avg_words_before': sum(len(str(text).split()) for text in original_texts) / len(original_texts),
            'avg_words_after': sum(len(text.split()) for text in cleaned_texts) / len(cleaned_texts),
            'length_reduction_pct': 0,
            'word_reduction_pct': 0
        }
        
        # Calculate reduction percentages
        if stats['avg_length_before'] > 0:
            stats['length_reduction_pct'] = (
                (stats['avg_length_before'] - stats['avg_length_after']) / 
                stats['avg_length_before'] * 100
            )
        
        if stats['avg_words_before'] > 0:
            stats['word_reduction_pct'] = (
                (stats['avg_words_before'] - stats['avg_words_after']) / 
                stats['avg_words_before'] * 100
            )
        
        return stats