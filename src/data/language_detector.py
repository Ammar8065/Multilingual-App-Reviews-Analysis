"""
Language detection and validation components.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from src.data.interfaces import LanguageDetectorInterface
from src.config import get_config
from src.utils.logger import get_logger


class LanguageDetector(LanguageDetectorInterface):
    """Language detector for validation and standardization of language codes."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Set seed for consistent results
        DetectorFactory.seed = 0
        
        # ISO 639-1 to ISO 639-2 mapping
        self.iso_639_mapping = {
            'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu', 'it': 'ita',
            'pt': 'por', 'ru': 'rus', 'zh': 'zho', 'ja': 'jpn', 'ko': 'kor',
            'ar': 'ara', 'hi': 'hin', 'th': 'tha', 'vi': 'vie', 'nl': 'nld',
            'sv': 'swe', 'da': 'dan', 'no': 'nor', 'fi': 'fin', 'pl': 'pol',
            'tr': 'tur', 'ms': 'msa', 'tl': 'tgl', 'id': 'ind', 'uk': 'ukr',
            'cs': 'ces', 'sk': 'slk', 'hu': 'hun', 'ro': 'ron', 'bg': 'bul',
            'hr': 'hrv', 'sr': 'srp', 'sl': 'slv', 'et': 'est', 'lv': 'lav',
            'lt': 'lit', 'mt': 'mlt', 'el': 'ell', 'he': 'heb', 'fa': 'fas',
            'ur': 'urd', 'bn': 'ben', 'ta': 'tam', 'te': 'tel', 'ml': 'mal',
            'kn': 'kan', 'gu': 'guj', 'pa': 'pan', 'or': 'ori', 'as': 'asm',
            'ne': 'nep', 'si': 'sin', 'my': 'mya', 'km': 'khm', 'lo': 'lao',
            'ka': 'kat', 'am': 'amh', 'sw': 'swa', 'zu': 'zul', 'af': 'afr',
            'sq': 'sqi', 'eu': 'eus', 'be': 'bel', 'bs': 'bos', 'ca': 'cat',
            'cy': 'cym', 'eo': 'epo', 'fo': 'fao', 'fy': 'fry', 'ga': 'gle',
            'gd': 'gla', 'gl': 'glg', 'is': 'isl', 'lb': 'ltz', 'mk': 'mkd',
            'mn': 'mon', 'nn': 'nno', 'nb': 'nob'
        }
        
        # Reverse mapping
        self.iso_639_reverse = {v: k for k, v in self.iso_639_mapping.items()}
        
        # Common language name mappings
        self.language_name_mapping = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'chinese': 'zh',
            'japanese': 'ja', 'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi',
            'thai': 'th', 'vietnamese': 'vi', 'dutch': 'nl', 'swedish': 'sv',
            'danish': 'da', 'norwegian': 'no', 'finnish': 'fi', 'polish': 'pl',
            'turkish': 'tr', 'malay': 'ms', 'tagalog': 'tl', 'indonesian': 'id'
        }
        
        # Language detection confidence threshold
        self.confidence_threshold = self.config.nlp.language_detection_threshold
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of the given text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Detected language code (ISO 639-1)
        """
        if not text or pd.isna(text) or len(str(text).strip()) < 3:
            return self.config.nlp.default_language
        
        try:
            # Clean text for better detection
            text = str(text).strip()
            
            # Use langdetect for detection
            detected_lang = detect(text)
            
            # Validate detected language
            if self.validate_language_code(detected_lang):
                return detected_lang
            else:
                self.logger.warning(f"Invalid language code detected: {detected_lang}")
                return self.config.nlp.default_language
                
        except LangDetectException as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return self.config.nlp.default_language
        except Exception as e:
            self.logger.error(f"Unexpected error in language detection: {str(e)}")
            return self.config.nlp.default_language
    
    def detect_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence score.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or pd.isna(text) or len(str(text).strip()) < 3:
            return self.config.nlp.default_language, 0.0
        
        try:
            text = str(text).strip()
            
            # Get language probabilities
            lang_probs = detect_langs(text)
            
            if lang_probs:
                best_lang = lang_probs[0]
                language_code = best_lang.lang
                confidence = best_lang.prob
                
                # Validate language code
                if self.validate_language_code(language_code):
                    return language_code, confidence
                else:
                    return self.config.nlp.default_language, 0.0
            else:
                return self.config.nlp.default_language, 0.0
                
        except Exception as e:
            self.logger.warning(f"Language detection with confidence failed: {str(e)}")
            return self.config.nlp.default_language, 0.0
    
    def validate_language_code(self, lang_code: str) -> bool:
        """
        Validate if language code is valid ISO 639-1 or ISO 639-2.
        
        Args:
            lang_code: Language code to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not lang_code or pd.isna(lang_code):
            return False
        
        lang_code = str(lang_code).lower().strip()
        
        # Check ISO 639-1 (2-letter codes)
        if len(lang_code) == 2 and lang_code in self.iso_639_mapping:
            return True
        
        # Check ISO 639-2 (3-letter codes)
        if len(lang_code) == 3 and lang_code in self.iso_639_reverse:
            return True
        
        # Check if it's in supported languages list
        if lang_code in self.config.nlp.supported_languages:
            return True
        
        return False
    
    def standardize_language_code(self, lang_code: str) -> str:
        """
        Standardize language code to ISO 639-1 format.
        
        Args:
            lang_code: Input language code
            
        Returns:
            Standardized language code (ISO 639-1)
        """
        if not lang_code or pd.isna(lang_code):
            return self.config.nlp.default_language
        
        lang_code = str(lang_code).lower().strip()
        
        # Already ISO 639-1
        if len(lang_code) == 2 and lang_code in self.iso_639_mapping:
            return lang_code
        
        # Convert from ISO 639-2 to ISO 639-1
        if len(lang_code) == 3 and lang_code in self.iso_639_reverse:
            return self.iso_639_reverse[lang_code]
        
        # Convert from language name
        if lang_code in self.language_name_mapping:
            return self.language_name_mapping[lang_code]
        
        # Handle common variations
        variations = {
            'zh-cn': 'zh', 'zh-tw': 'zh', 'zh-hk': 'zh',
            'en-us': 'en', 'en-gb': 'en', 'en-au': 'en',
            'es-es': 'es', 'es-mx': 'es', 'es-ar': 'es',
            'fr-fr': 'fr', 'fr-ca': 'fr',
            'pt-br': 'pt', 'pt-pt': 'pt',
            'de-de': 'de', 'de-at': 'de', 'de-ch': 'de'
        }
        
        if lang_code in variations:
            return variations[lang_code]
        
        # Extract base language from locale codes
        if '-' in lang_code:
            base_lang = lang_code.split('-')[0]
            if self.validate_language_code(base_lang):
                return base_lang
        
        if '_' in lang_code:
            base_lang = lang_code.split('_')[0]
            if self.validate_language_code(base_lang):
                return base_lang
        
        # If all else fails, return default
        self.logger.warning(f"Could not standardize language code: {lang_code}")
        return self.config.nlp.default_language
    
    def standardize_language_codes(self, df: pd.DataFrame, 
                                  language_column: str = 'review_language') -> pd.DataFrame:
        """
        Standardize language codes in a DataFrame.
        
        Args:
            df: Input DataFrame
            language_column: Name of the language column
            
        Returns:
            DataFrame with standardized language codes
        """
        if language_column not in df.columns:
            self.logger.error(f"Language column '{language_column}' not found in DataFrame")
            return df
        
        df_processed = df.copy()
        
        self.logger.info(f"Standardizing language codes in column '{language_column}'")
        
        # Standardize language codes
        original_codes = df_processed[language_column].fillna('').tolist()
        standardized_codes = [self.standardize_language_code(code) for code in original_codes]
        
        df_processed[f'{language_column}_standardized'] = standardized_codes
        
        # Add validation results
        validation_results = [self.validate_language_code(code) for code in original_codes]
        df_processed[f'{language_column}_valid'] = validation_results
        
        # Statistics
        unique_original = set(original_codes)
        unique_standardized = set(standardized_codes)
        invalid_count = sum(1 for valid in validation_results if not valid)
        
        self.logger.info(f"Language standardization completed:")
        self.logger.info(f"  - Original unique languages: {len(unique_original)}")
        self.logger.info(f"  - Standardized unique languages: {len(unique_standardized)}")
        self.logger.info(f"  - Invalid codes found: {invalid_count}")
        
        return df_processed
    
    def detect_and_validate_languages(self, df: pd.DataFrame, 
                                    text_column: str = 'review_text',
                                    language_column: str = 'review_language') -> pd.DataFrame:
        """
        Detect languages from text and compare with stated languages.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            language_column: Name of the stated language column
            
        Returns:
            DataFrame with detected languages and validation results
        """
        if text_column not in df.columns:
            self.logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df
        
        df_processed = df.copy()
        
        self.logger.info(f"Detecting languages from text in column '{text_column}'")
        
        # Detect languages from text
        texts = df_processed[text_column].fillna('').tolist()
        detected_languages = []
        detection_confidences = []
        
        for text in texts:
            lang, confidence = self.detect_language_with_confidence(text)
            detected_languages.append(lang)
            detection_confidences.append(confidence)
        
        df_processed['detected_language'] = detected_languages
        df_processed['detection_confidence'] = detection_confidences
        
        # Compare with stated languages if available
        if language_column in df.columns:
            stated_languages = df_processed[language_column].fillna('').tolist()
            standardized_stated = [self.standardize_language_code(lang) for lang in stated_languages]
            
            # Check agreement between detected and stated languages
            language_agreement = []
            for detected, stated in zip(detected_languages, standardized_stated):
                agreement = detected == stated
                language_agreement.append(agreement)
            
            df_processed['language_agreement'] = language_agreement
            df_processed[f'{language_column}_standardized'] = standardized_stated
            
            # Statistics
            agreement_rate = sum(language_agreement) / len(language_agreement) * 100
            high_confidence_agreement = sum(
                1 for agree, conf in zip(language_agreement, detection_confidences)
                if agree and conf >= self.confidence_threshold
            ) / len(language_agreement) * 100
            
            self.logger.info(f"Language detection and validation completed:")
            self.logger.info(f"  - Overall agreement rate: {agreement_rate:.2f}%")
            self.logger.info(f"  - High confidence agreement: {high_confidence_agreement:.2f}%")
            self.logger.info(f"  - Average detection confidence: {sum(detection_confidences)/len(detection_confidences):.3f}")
        
        return df_processed
    
    def get_language_statistics(self, df: pd.DataFrame, 
                              language_column: str = 'review_language') -> Dict[str, any]:
        """
        Generate language distribution statistics.
        
        Args:
            df: Input DataFrame
            language_column: Name of the language column
            
        Returns:
            Dictionary with language statistics
        """
        if language_column not in df.columns:
            self.logger.error(f"Language column '{language_column}' not found")
            return {}
        
        # Language distribution
        lang_counts = df[language_column].value_counts()
        lang_percentages = df[language_column].value_counts(normalize=True) * 100
        
        # Validation statistics if available
        validation_stats = {}
        if f'{language_column}_valid' in df.columns:
            valid_count = df[f'{language_column}_valid'].sum()
            validation_stats = {
                'valid_codes': int(valid_count),
                'invalid_codes': len(df) - int(valid_count),
                'validation_rate': (valid_count / len(df)) * 100
            }
        
        # Detection agreement statistics if available
        agreement_stats = {}
        if 'language_agreement' in df.columns:
            agreement_count = df['language_agreement'].sum()
            agreement_stats = {
                'agreements': int(agreement_count),
                'disagreements': len(df) - int(agreement_count),
                'agreement_rate': (agreement_count / len(df)) * 100
            }
        
        # Confidence statistics if available
        confidence_stats = {}
        if 'detection_confidence' in df.columns:
            confidences = df['detection_confidence'].dropna()
            confidence_stats = {
                'avg_confidence': float(confidences.mean()),
                'min_confidence': float(confidences.min()),
                'max_confidence': float(confidences.max()),
                'high_confidence_count': int((confidences >= self.confidence_threshold).sum()),
                'high_confidence_rate': float((confidences >= self.confidence_threshold).mean() * 100)
            }
        
        statistics = {
            'total_records': len(df),
            'unique_languages': len(lang_counts),
            'language_distribution': {
                'counts': lang_counts.to_dict(),
                'percentages': {k: round(v, 2) for k, v in lang_percentages.to_dict().items()}
            },
            'most_common_language': lang_counts.index[0] if not lang_counts.empty else None,
            'least_common_languages': lang_counts.tail(5).index.tolist(),
            'validation_statistics': validation_stats,
            'agreement_statistics': agreement_stats,
            'confidence_statistics': confidence_stats
        }
        
        return statistics
    
    def recommend_language_corrections(self, df: pd.DataFrame,
                                     text_column: str = 'review_text',
                                     language_column: str = 'review_language') -> List[Dict]:
        """
        Recommend corrections for language mismatches.
        
        Args:
            df: DataFrame with language detection results
            text_column: Name of the text column
            language_column: Name of the language column
            
        Returns:
            List of correction recommendations
        """
        recommendations = []
        
        if 'language_agreement' not in df.columns or 'detection_confidence' not in df.columns:
            self.logger.warning("Language detection results not found. Run detect_and_validate_languages first.")
            return recommendations
        
        # Find high-confidence disagreements
        disagreements = df[
            (~df['language_agreement']) & 
            (df['detection_confidence'] >= self.confidence_threshold)
        ]
        
        for idx, row in disagreements.iterrows():
            recommendation = {
                'index': idx,
                'current_language': row.get(language_column, ''),
                'detected_language': row.get('detected_language', ''),
                'confidence': row.get('detection_confidence', 0.0),
                'text_sample': str(row.get(text_column, ''))[:100] + '...',
                'recommendation': f"Consider changing from '{row.get(language_column, '')}' to '{row.get('detected_language', '')}'"
            }
            recommendations.append(recommendation)
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.logger.info(f"Generated {len(recommendations)} language correction recommendations")
        
        return recommendations[:50]  # Limit to top 50 recommendations