import os
import sys
import re
import joblib
from src.preprocess import clean_text

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# CONTEXT-DEPENDENT WORD DICTIONARY (Dual Meanings)
# ============================================================
CONTEXT_WORDS = {
    'ghanta': {
        'safe_keywords': ['mandir', 'temple', 'pooja', 'aarti', 'bajana', 'baj', 'bajao', 'bajaya',
                          'bajta', 'bajte', 'bajate', 'church', 'school', 'awaaz', 'bell', 'sundar',
                          'ek', 'do', 'teen', 'char', 'paanch', 'chhe', 'saat', 'aath', 'nau', 'das',
                          '1', '2', '3', '4', '5', 'lagega', 'lagenge', 'samay', 'waqt', 'hour', 'hours', 'der', 'aadha', 'kitne', 'kitna'],
        'safe_label': 'Normal',
        'safe_reason': '🔔 "Ghanta" detected in religious/bell/time context (temple bell or time duration) — classified as Normal.',
        'offensive_reason': '⚠️ "Ghanta" detected as vulgar slang (dismissive/offensive usage).'
    },
    'ghante': {
        'safe_keywords': ['mandir', 'temple', 'pooja', 'aarti', 'bajana', 'baj', 'bajao', 'bajaya',
                          'bajta', 'bajte', 'bajate', 'church', 'school', 'awaaz', 'bell',
                          'ek', 'do', 'teen', 'char', 'paanch', 'chhe', 'saat', 'aath', 'nau', 'das',
                          '1', '2', '3', '4', '5', 'lagega', 'lagenge', 'samay', 'waqt', 'hour', 'hours', 'der', 'aadha', 'kitne', 'kitna', 'baad', 'pehle'],
        'safe_label': 'Normal',
        'safe_reason': '⏱️ "Ghante" detected in time/duration context — classified as Normal.',
        'offensive_reason': '⚠️ "Ghante" detected in offensive/vulgar context.'
    },
    'घंटा': {
        'safe_keywords': ['मंदिर', 'पूजा', 'आरती', 'बजाना', 'बज', 'बजाओ', 'बजाया', 'बजता', 'बजते', 'स्कूल', 'आवाज', 'सुंदर', 'घंटी',
                          'एक', 'दो', 'तीन', 'चार', 'पांच', 'समय', 'वक्त', 'लगेगा', 'लगेगी', 'लगेंगे', 'आधा', 'देर', 'कितने', 'कितना'],
        'safe_label': 'Normal',
        'safe_reason': '🔔 "घंटा" religious/bell/time context (घंटी या समय) mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "घंटा" vulgar slang ke roop mein paya gaya.'
    },
    'घंटे': {
        'safe_keywords': ['मंदिर', 'पूजा', 'आरती', 'बजाना', 'बज', 'बजाओ', 'बजाया', 'बजता', 'बजते', 'स्कूल', 'आवाज', 'सुंदर', 'घंटी',
                          'एक', 'दो', 'तीन', 'चार', 'पांच', 'समय', 'वक्त', 'लगेगा', 'लगेगी', 'लगेंगे', 'आधा', 'देर', 'कितने', 'बाद', 'पहले'],
        'safe_label': 'Normal',
        'safe_reason': '⏱️ "घंटे" time/duration context (समय) mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "घंटे" vulgar slang ke roop mein paya gaya.'
    },
    'kutta': {
        'safe_keywords': ['mera', 'pyaara', 'ghar', 'rakhwala', 'doctor', 'desi', 'samajhdaar',
                          'mohalle', 'khel', 'street', 'khana', 'bacche', 'pet', 'paaltu', 'cute'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "Kutta" detected in pet/animal context (talking about a dog) — classified as Normal.',
        'offensive_reason': '⚠️ "Kutta" detected as targeted insult (calling someone a dog).'
    },
    'कुत्ता': {
        'safe_keywords': ['मेरा', 'प्यारा', 'घर', 'रखवाला', 'डॉक्टर', 'देसी', 'समझदार', 'मोहल्ले', 'खेल', 'खाना', 'बच्चे', 'पालतू'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "कुत्ता" pet/animal context mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "कुत्ता" insult ke roop mein use hua.'
    },
    'kutte': {
        'safe_keywords': ['mera', 'pyaara', 'ghar', 'rakhwala', 'doctor', 'desi', 'samajhdaar',
                          'mohalle', 'khel', 'street', 'khana', 'bacche', 'pet', 'paaltu'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "Kutte" detected in pet/animal context — classified as Normal.',
        'offensive_reason': '⚠️ "Kutte" detected as targeted insult.'
    },
    'कुत्ते': {
        'safe_keywords': ['मेरा', 'प्यारा', 'घर', 'रखवाला', 'डॉक्टर', 'देसी', 'समझदार', 'मोहल्ले', 'खेल', 'खाना', 'बच्चे', 'पालतू'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "कुत्ते" animal context mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "कुत्ते" insult ke roop mein use hua.'
    },
    'saala': {
        'safe_keywords': ['mera', 'behen', 'wife', 'bhai', 'shaadi', 'ghar', 'family', 'rishtedaar',
                          'sahab', 'milke', 'accha', 'insaan', 'help', 'matlab', 'rehta'],
        'safe_label': 'Normal',
        'safe_reason': '👨‍👩‍👦 "Saala" detected in family context (brother-in-law) — classified as Normal.',
        'offensive_reason': '⚠️ "Saala" detected as abusive slang (targeted insult).'
    },
    'साला': {
        'safe_keywords': ['मेरा', 'बहन', 'पत्नी', 'भाई', 'शादी', 'घर', 'परिवार', 'रिश्तेदार', 'साहब', 'अच्छा'],
        'safe_label': 'Normal',
        'safe_reason': '👨‍👩‍👦 "साला" family relation (brother-in-law) mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "साला" gaali ke roop mein use hua.'
    },
    'saale': {
        'safe_keywords': ['mera', 'behen', 'wife', 'bhai', 'shaadi', 'ghar', 'family', 'rishtedaar',
                          'sahab', 'milke', 'accha', 'help'],
        'safe_label': 'Normal',
        'safe_reason': '👨‍👩‍👦 "Saale" detected in family context (brother-in-law) — classified as Normal.',
        'offensive_reason': '⚠️ "Saale" detected as abusive slang.'
    },
    'साले': {
        'safe_keywords': ['मेरा', 'बहन', 'पत्नी', 'भाई', 'शादी', 'घर', 'परिवार', 'रिश्तेदार', 'साहब'],
        'safe_label': 'Normal',
        'safe_reason': '👨‍👩‍👦 "साले" family relation context mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "साले" abusive context mein paya gaya.'
    },
    'gadha': {
        'safe_keywords': ['gaon', 'jaanwar', 'animal', 'sawari', 'village', 'mehnati', 'baccho', 'farm', 'khet'],
        'safe_label': 'Normal',
        'safe_reason': '🫏 "Gadha" detected in animal context (donkey) — classified as Normal.',
        'offensive_reason': '⚠️ "Gadha" used as insult.'
    },
    'गधा': {
        'safe_keywords': ['गांव', 'जानवर', 'सवारी', 'मेहनती', 'खेत', 'जंगल'],
        'safe_label': 'Normal',
        'safe_reason': '🫏 "गधा" animal context mein paya gaya — classified as Normal.',
        'offensive_reason': '⚠️ "गधा" insult ke roop mein use hua.'
    }
}

# ============================================================
# TIER 1: SEVERE HATE SPEECH & VIOLENT THREATS DATABASE
# ============================================================
EXPLICIT_HATE_SPEECH = {
    # Devanagari Violent & Lethal Threats
    "मार दूंगा", "मार डालूंगा", "जान से मार", "खत्म कर दूंगा", "गोली मार", "काट दूंगा",
    "उड़ा दूंगा", "ज़िंदा जला", "जिंदा जला", "चीर दूंगा", "फाड़ दूंगा", "गला काट", "तबाह कर दूंगा",
    
    # Roman / Hinglish Violent Threats
    "maar dunga", "maar dalunga", "jaan se maar", "khatam kar dunga", "goli maar",
    "kaat dunga", "uda dunga", "zinda jala", "cheer dunga", "tabah kar dunga",
    
    # English Violent Threats & Hate
    "i will kill you", "will kill you", "kill you scum", "murder you", "shoot you",
    "cut your throat", "burn you", "burn to death", "die you absolute trash",
    "kill yourself", "die you scum",
    
    # Communal / Identity / Terror Hate Markers
    "आतंकवादी", "जिहादी", "गद्दार", "देशद्रोही", "काफिर", "chakka", "hijra", "katue",
    "terrorist", "nazi", "faggot", "faggots", "kill all"
}

# ============================================================
# TIER 2: CASUAL PROFANITY & OFFENSIVE SLANG DATABASE (Multi-Script)
# ============================================================
EXPLICIT_OFFENSIVE = {
    # Devanagari Hindi Profanities & Informal Slang
    "हरामज़ादे", "हरामजादे", "हरामजादा", "हरामज़ादा", "हरामी", "हरामखोर",
    "भोसड़ीके", "भोसड़ीके", "भोसडीके", "भोसडे", "भोसड़ा", "भोसुडी", "भोस्डी", "भोसड़ी",
    "मादरचोद", "मदरचोद", "मादरजात", "बहनचोद", "बहेनचोद", "बेहेनचोद",
    "चूतिया", "चूतिये", "चूतियापा", "चूत", "कमीना", "कमीने", "कमीनी",
    "लौड़े", "लौंडे", "लंड", "लौड़ा", "लवड़े", "लवड़ा", "लावडे", "लावडा",
    "गांडू", "गांड", "रांडी", "रंडी", "झांटू", "झांट", "भाड़वे", "भड़वे", "भड़वा",
    "सूअर", "चोदू", "चोमू", "दलाल", "रंडवा",
    
    # Roman / Hinglish Profanities & Slangs
    "haramzade", "haramzada", "haramzadi", "harami", "haramkhor",
    "bhosdike", "bhosdika", "bhosadi", "bhosadike", "bsdk", "bhosdk", "bhosdi", "boshde",
    "madarchod", "maderchod", "mc", "behenchod", "bhenchod", "bc", "bhenkelaude",
    "chutiya", "chutiye", "chut", "chutiyapa", "gaand", "gandu", "kamina", "kamine",
    "laude", "lodu", "loda", "lund", "lavde", "lavda", "randi", "jhantu", "jhant", "bhadwe",
    
    # English Slurs / Profanities
    "fuck", "fucking", "fcker", "bitch", "bastard", "asshole", "whore", "slut", "cunt",
    "moron", "retarded", "retarted"
}


class HateSpeechPredictor:
    def __init__(self, models_dir='models'):
        model_path = os.path.join(models_dir, 'model.pkl')
        vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or Vectorizer not found. Please train the model first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    
    def detect_context(self, text):
        text_lower = text.lower()
        words_in_text = set(re.findall(r'[\w\u0900-\u097F]+', text_lower))
        
        for context_word, config in CONTEXT_WORDS.items():
            if context_word in words_in_text or context_word in text_lower:
                safe_hits = words_in_text.intersection(set(config['safe_keywords']))
                if safe_hits:
                    return {
                        'found_word': context_word,
                        'is_safe': True,
                        'override_label': config['safe_label'],
                        'reason': config['safe_reason'],
                        'context_keywords': list(safe_hits)
                    }
                else:
                    return {
                        'found_word': context_word,
                        'is_safe': False,
                        'override_label': None,
                        'reason': config['offensive_reason'],
                        'context_keywords': []
                    }
        
        return None

    def detect_hate_threats(self, text):
        text_lower = text.lower()
        matched = []
        for threat in EXPLICIT_HATE_SPEECH:
            if threat in text_lower:
                matched.append(threat)
        return matched

    def detect_offensive_slurs(self, text):
        text_lower = text.lower()
        matched = []
        for slur in EXPLICIT_OFFENSIVE:
            if slur in text_lower:
                matched.append(slur)
        return matched

    def predict(self, text):
        cleaned_text = clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        
        label = str(prediction)
        
        # 1. Check Context Overrides (Religious/Animal/Family relation)
        context_info = self.detect_context(text)
        if context_info and context_info.get('is_safe'):
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'prediction': 'Normal',
                'context': context_info
            }

        # 2. Check Explicit Severe Hate Speech & Violent Threats
        matched_hate = self.detect_hate_threats(text)
        if matched_hate:
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'prediction': 'Hate Speech',
                'context': {
                    'found_word': matched_hate[0],
                    'is_safe': False,
                    'category': 'Hate Speech',
                    'reason': f"🚨 Violent Threat / Hate Speech detected: '{matched_hate[0]}'"
                }
            }

        # 3. Check Explicit Casual Abusive / Profanities (e.g. हरामजादे, चूतिया)
        matched_offensive = self.detect_offensive_slurs(text)
        if matched_offensive:
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'prediction': 'Offensive',
                'context': {
                    'found_word': matched_offensive[0],
                    'is_safe': False,
                    'category': 'Offensive Language',
                    'reason': f"⚠️ Casual / Abusive Slang detected: '{matched_offensive[0]}' (Informal profanity, not hate speech)"
                }
            }

        # 4. Context words in non-safe context (e.g. 'ghanta' without bell context)
        if context_info and not context_info.get('is_safe'):
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'prediction': 'Offensive',
                'context': context_info
            }

        # 5. Fallback to trained Machine Learning Model ('Normal', 'Offensive', 'Hate Speech')
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'prediction': label,
            'context': None
        }

if __name__ == "__main__":
    predictor = HateSpeechPredictor()
    tests = [
        "आयुष हरामजादे पढ़ लिया कर यार पढ़ लिया",
        "Mandir mein ghanta baj raha hai",
        "Ghanta farak padta hai tujhe",
        "Mera kutta bahut pyaara hai",
        "Kutte kamine jaan se maar dunga tujhe",
        "नमस्ते, आप कैसे हैं?"
    ]
    for t in tests:
        result = predictor.predict(t)
        ctx = result.get('context')
        ctx_info = f" | Context: {ctx['reason']}" if ctx else ""
        print(f"Text: {t} -> {result['prediction']}{ctx_info}")
