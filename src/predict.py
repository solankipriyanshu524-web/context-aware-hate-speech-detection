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
    },
    'ullu': {
        'safe_keywords': ['raat', 'ped', 'tree', 'pakshi', 'bird', 'wildlife', 'jungle', 'forest', 'aakhein', 'nature'],
        'safe_label': 'Normal',
        'safe_reason': '🦉 "Ullu" detected in nocturnal bird/wildlife context (owl) — classified as Normal.',
        'offensive_reason': '⚠️ "Ullu" detected as slang insult (calling someone a fool).'
    },
    'उल्लू': {
        'safe_keywords': ['रात', 'पेड़', 'पक्षी', 'चिड़िया', 'जंगल', 'वन्यजीव', 'प्रकृति', 'आंखें'],
        'safe_label': 'Normal',
        'safe_reason': '🦉 "उल्लू" पक्षी / वन्यजीव के संदर्भ में पाया गया — classified as Normal.',
        'offensive_reason': '⚠️ "उल्लू" मूर्ख / बेवकूफ बनाने के स्लैंग में प्रयोग हुआ।'
    },
    'pagal': {
        'safe_keywords': ['cricket', 'padhai', 'shauk', 'passion', 'hospital', 'doctor', 'ilaj', 'mehnat', 'deewana', 'geet', 'khel'],
        'safe_label': 'Normal',
        'safe_reason': '🧠 "Pagal" detected in passion/enthusiasm/medical context — classified as Normal.',
        'offensive_reason': '⚠️ "Pagal" detected as derogatory mental insult.'
    },
    'पागल': {
        'safe_keywords': ['क्रिकेट', 'पढ़ाई', 'शौक', 'जुनून', 'अस्पताल', 'डॉक्टर', 'इलाज', 'मेहनत', 'दीवाना', 'गीत', 'खेल'],
        'safe_label': 'Normal',
        'safe_reason': '🧠 "पागल" जुनून या मेडिकल संदर्भ में पाया गया — classified as Normal.',
        'offensive_reason': '⚠️ "पागल" अपमानजनक मानसिक कटाक्ष के रूप में प्रयुक्त हुआ।'
    },
    'kamina': {
        'safe_keywords': ['yaar', 'dost', 'friend', 'kitne din', 'masti', 'bhai', 'kaisa', 'college', 'school', 'bachpan'],
        'safe_label': 'Normal',
        'safe_reason': '🤝 "Kamina" detected in informal friend banter/camaraderie — classified as Normal.',
        'offensive_reason': '⚠️ "Kamina" detected as malicious character insult.'
    },
    'कमीना': {
        'safe_keywords': ['यार', 'दोस्त', 'भाई', 'कितने दिन', 'मस्ती', 'कॉलेज', 'स्कूल', 'बचपन', 'कैसा'],
        'safe_label': 'Normal',
        'safe_reason': '🤝 "कमीना" दोस्तों के बीच अनौपचारिक दोस्ताना हंसी-मजाक में पाया गया — classified as Normal.',
        'offensive_reason': '⚠️ "कमीना" व्यक्तिगत द्वेष या गाली के रूप में प्रयुक्त हुआ।'
    },
    'billi': {
        'safe_keywords': ['pet', 'paaltu', 'doodh', 'ghar', 'cute', 'animal', 'kitten', 'white', 'black', 'meow'],
        'safe_label': 'Normal',
        'safe_reason': '🐱 "Billi" detected in domestic cat context — classified as Normal.',
        'offensive_reason': '⚠️ "Billi" detected in derogatory context.'
    },
    'बिल्ली': {
        'safe_keywords': ['पालतू', 'दूध', 'घर', 'सफेद', 'काली', 'प्यारी', 'जानवर'],
        'safe_label': 'Normal',
        'safe_reason': '🐱 "बिल्ली" पालतू जानवर के संदर्भ में पाया गया — classified as Normal.',
        'offensive_reason': '⚠️ "बिल्ली" आपत्तिजनक रूप में प्रयुक्त हुआ।'
    },
    # English Context Words Disambiguation
    'bitch': {
        'safe_keywords': ['female', 'dog', 'puppy', 'puppies', 'litter', 'breeder', 'breeding', 'canine', 'animal', 'vet', 'veterinary'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "Bitch" detected in biological canine / breeding context (female dog) — classified as Normal.',
        'offensive_reason': '⚠️ "Bitch" detected as misogynistic / targeted insult.'
    },
    'kill': {
        'safe_keywords': ['stage', 'performance', 'show', 'song', 'dance', 'laughing', 'time', 'joke', 'awesome', 'crushed', 'vibe', 'killing'],
        'safe_label': 'Normal',
        'safe_reason': '⚡ "Kill" detected as colloquial performance praise / positive metaphor ("killing it") — classified as Normal.',
        'offensive_reason': '🚨 "Kill" detected in violent threat context.'
    },
    'shoot': {
        'safe_keywords': ['video', 'photo', 'film', 'camera', 'scene', 'movie', 'hoops', 'basketball', 'goal', 'picture', 'commercial'],
        'safe_label': 'Normal',
        'safe_reason': '🎥 "Shoot" detected in camera / video production or sports context — classified as Normal.',
        'offensive_reason': '🚨 "Shoot" detected in violent firearm threat context.'
    },
    'pig': {
        'safe_keywords': ['farm', 'animal', 'swine', 'piglet', 'piglets', 'mud', 'barn', 'agriculture', 'livestock'],
        'safe_label': 'Normal',
        'safe_reason': '🐖 "Pig" detected in farm animal / livestock context — classified as Normal.',
        'offensive_reason': '⚠️ "Pig" detected as derogatory insult.'
    },
    'bomb': {
        'safe_keywords': ['the', 'movie', 'party', 'track', 'album', 'awesome', 'cool', 'song'],
        'safe_label': 'Normal',
        'safe_reason': '💥 "The bomb" detected as informal slang for great / impressive — classified as Normal.',
        'offensive_reason': '🚨 "Bomb" detected in explosive terror threat context.'
    },
    'sick': {
        'safe_keywords': ['trick', 'move', 'skate', 'beat', 'music', 'game', 'cool', 'awesome', 'style', 'guitar', 'drop'],
        'safe_label': 'Normal',
        'safe_reason': '🛹 "Sick" detected as modern slang for impressive / awesome — classified as Normal.',
        'offensive_reason': '⚠️ "Sick" detected in derogatory / harassment context.'
    },
    # Telugu Context Words Disambiguation (Telugu Script & Romanized Tenglish)
    'kukka': {
        'safe_keywords': ['maa', 'intlo', 'chala', 'baguntundi', 'pet', 'cute', 'pillalu', 'pedda', 'chinna', 'illu', 'intiki', 'pempudu', 'prema'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "Kukka" detected in domestic pet context in Telugu (కుక్క / Pet Dog) — classified as Normal.',
        'offensive_reason': '⚠️ "Kukka" detected as targeted abusive insult in Telugu.'
    },
    'కుక్క': {
        'safe_keywords': ['మా', 'ఇంట్లో', 'చాలా', 'బాగుంటుంది', 'పెట్', 'ముద్దుగా', 'పిల్లలు', 'పెద్ద', 'చిన్న', 'రక్షణ', 'పెంపుడు', 'ఇంటి'],
        'safe_label': 'Normal',
        'safe_reason': '🐕 "కుక్క" domestic pet context (పెంపుడు కుక్క) లో కనుగొనబడింది — classified as Normal.',
        'offensive_reason': '⚠️ "కుక్క" దూషణ / అవమానకరమైన పదంగా ఉపయోగించబడింది.'
    },
    'pichi': {
        'safe_keywords': ['cricket', 'cinema', 'songs', 'chala', 'istam', 'prema', 'movie', 'abhimanam', 'hero'],
        'safe_label': 'Normal',
        'safe_reason': '🏏 "Pichi" detected in passionate fandom / hobby context in Telugu (cricket/cinema craze) — Normal.',
        'offensive_reason': '⚠️ "Pichi" detected as derogatory mental insult in Telugu.'
    },
    'పిచ్చి': {
        'safe_keywords': ['క్రికెట్', 'సినిమా', 'పాటలు', 'చాలా', 'ఇష్టం', 'ప్రేమ', 'అభిమానం', 'హీరో'],
        'safe_label': 'Normal',
        'safe_reason': '🏏 "పిచ్చి" క్రీడాభిమానం లేదా హాబీ సందర్భంలో ఉపయోగించబడింది — classified as Normal.',
        'offensive_reason': '⚠️ "పిచ్చి" మానసిక అవమానకరమైన పదంగా ఉపయోగించబడింది.'
    },
    'donga': {
        'safe_keywords': ['papa', 'babu', 'chinna', 'navvu', 'muddhu', 'pillalu', 'kanna', 'bujji'],
        'safe_label': 'Normal',
        'safe_reason': '👶 "Donga" detected as affectionate playful teasing of a child in Telugu — classified as Normal.',
        'offensive_reason': '⚠️ "Donga" detected as criminal thief / derogatory insult in Telugu.'
    },
    'దొంగ': {
        'safe_keywords': ['పాప', 'బాబు', 'చిన్న', 'నవ్వు', 'ముద్దు', 'పిల్లలు', 'కన్నా', 'బుజ్జి'],
        'safe_label': 'Normal',
        'safe_reason': '👶 "దొంగ" చిన్న పిల్లల ముద్దు పేచీ లేదా నవ్వు సందర్భంలో వాడబడింది — classified as Normal.',
        'offensive_reason': '⚠️ "దొంగ" నేరపూరిత దూషణగా వాడబడింది.'
    },
    'gadida': {
        'safe_keywords': ['polam', 'pashuvu', 'pani', 'baruvu', 'janthuvu', 'kasta', 'rythu'],
        'safe_label': 'Normal',
        'safe_reason': '🫏 "Gadida" detected in hardworking farm animal context in Telugu — classified as Normal.',
        'offensive_reason': '⚠️ "Gadida" detected as intellectual ridicule / insult in Telugu.'
    },
    'గాడిద': {
        'safe_keywords': ['పొలం', 'పశువు', 'పని', 'బరువు', 'జంతువు', 'రైతు'],
        'safe_label': 'Normal',
        'safe_reason': '🫏 "గాడిద" శ్రమించే జంతువు సందర్భంలో గుర్తించబడింది — classified as Normal.',
        'offensive_reason': '⚠️ "గాడిద" బుద్ధిహీనతను కించపరిచే తిట్టుగా వాడబడింది.'
    }
}

# ============================================================
# TIER 1: SEVERE HATE SPEECH & VIOLENT THREATS DATABASE (Multilingual)
# ============================================================
EXPLICIT_HATE_SPEECH = {
    # Devanagari Violent & Lethal Threats (Hindi/Marathi/Phonetic English)
    "मार दूंगा", "मार डालूंगा", "जान से मार", "खत्म कर दूंगा", "गोली मार", "काट दूंगा",
    "उड़ा दूंगा", "ज़िंदा जला", "जिंदा जला", "चीर दूंगा", "फाड़ दूंगा", "गला काट", "तबाह कर दूंगा",
    "मारून टाकीन", "जीव घेईन",
    # Phonetic English in Devanagari (from speech-to-text or typing)
    "आई वांट टू किल यू", "वांट टू किल यू", "आई विल किल यू", "विल किल यू", "किल यू", "शूट यू", "मर्डर यू", "किल कर दूंगा",
    
    # Roman / Hinglish Violent Threats (with single & double 'a')
    "maar dunga", "mar dunga", "maar dalunga", "mar dalunga", "jaan se maar", "jaan se mar",
    "jaan se maar dunga", "jaan se mar dunga", "tujhe maar dunga", "tujhe mar dunga",
    "khatam kar dunga", "khatm kar dunga", "goli maar", "goli mar", "marunga", "maarunga",
    "kaat dunga", "uda dunga", "zinda jala", "cheer dunga", "tabah kar dunga",
    
    # Telugu Violent Threats (Telugu Script & Tenglish)
    "చంపేస్తా", "చంపేస్తాను", "నిన్ను చంపేస్తా", "నరికేస్తా", "నరికేస్తాను", "ప్రాణం తీస్తా", "ఖతం చేస్తా", "రక్తం తాగుతా",
    "champesta", "champestanu", "ninnu champesta", "narikesta", "narikestanu", "pranam teesta", "katham chesta", "raktam taagutha",
    
    # Tamil & Bengali Violent Threats
    "கொன்னுடுவேன்", "kondruven", "মেরে ফেলবো", "mere phelbo",
    
    # English Violent Threats & Hate
    "i will kill you", "will kill you", "i want to kill you", "want to kill you", "kill you scum", "kill you", "murder you", "shoot you",
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
    
    # Telugu Profanities & Slangs (Telugu Script & Tenglish)
    "లంచోడు", "దొంగ నా కొడకా", "దవడ పగలగొడతా", "వెధవ", "నీ యమ్మ", "నీ అమ్మ", "ముండ", "బోకే", "లంజ", "గుద్ద",
    "vedhava", "naa kodaka", "donga na kodaka", "lanja", "lanjamunda", "mundamopi", "boke", "nee amma", "gudha", "dengey",
    
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
        words_in_text = set(re.findall(r'[\w\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0980-\u09FF]+', text_lower))
        
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
        
        # Robust Regex Patterns for flexible Hinglish, Hindi, Telugu, and English death threats
        if not matched:
            patterns = [
                r'(?:tujhe|tumhe|tere\s*ko|tera)?\s*(?:jaan\s*s[e|y]\s*)?m[a]+r\s*(?:d[u|o]ng[a|e|i]|d[a]*lung[a|e|i])',
                r'm[a]+rung[a|e|i]',
                r'khat[a]*m\s*kar\s*d[u|o]ng[a|e|i]',
                r'goli\s*m[a]+r',
                r'uda\s*d[u|o]ng[a|e|i]',
                r'(?:i\s+)?(?:want\s+to|will|gonna|going\s+to)\s+kill\s+you',
                r'kill\s+you',
                r'आई\s*वांट\s*टू\s*किल\s*यू|वांट\s*टू\s*किल|किल\s*यू|आई\s*विल\s*किल\s*यू|शूट\s*यू',
                r'(?:ninnu\s+)?champ[e|a]st[a|u]|narik[e|a]st[a|u]|pranam\s*teest[a|u]',
                r'చంపేస్తా|నరికేస్తా|ప్రాణం\s*తీస్తా'
            ]
            for pat in patterns:
                m = re.search(pat, text_lower, re.IGNORECASE)
                if m:
                    matched.append(m.group(0))
                    break
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
