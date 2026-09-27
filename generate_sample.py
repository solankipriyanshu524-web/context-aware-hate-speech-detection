import pandas as pd
import os

def create_sample_data():
    os.makedirs('data', exist_ok=True)
    
    data = [
        # Explicitly Hate Speech / Offensive (English)
        {"text": "Asshole, learn how to drive!", "label": "Offensive"},
        {"text": "You faggots are ruining everything.", "label": "Hate Speech"},
        {"text": "What a retarded thing to say.", "label": "Offensive"},
        {"text": "I will kill you scum.", "label": "Hate Speech"},
        {"text": "You are a dirty bitch.", "label": "Hate Speech"},
        {"text": "Die you absolute trash.", "label": "Hate Speech"},
        {"text": "Shut up moron.", "label": "Offensive"},
        {"text": "You stupid idiot why did you do that?", "label": "Offensive"},
        {"text": "Go to hell you bastard.", "label": "Hate Speech"},
        {"text": "I hate you with all my heart.", "label": "Hate Speech"},
        {"text": "You are a total loser.", "label": "Offensive"},
        {"text": "Worthless piece of shit.", "label": "Hate Speech"},
        {"text": "Fucking whore.", "label": "Hate Speech"},
        
        # Explicitly Hate Speech / Offensive (Hindi/Hinglish)
        {"text": "Tu chutiya hai kya?", "label": "Offensive"},
        {"text": "Madarchod, tujhe zinda nahi chhodunga.", "label": "Hate Speech"},
        {"text": "Bhenchod kitna dimag kharab karega tu.", "label": "Offensive"},
        {"text": "Kutte kamine, tabaah kar dunga tujhe.", "label": "Hate Speech"},
        {"text": "Gandu aadmi hai tu ek number ka.", "label": "Offensive"},
        {"text": "Randwa saala, sharam nahi aati.", "label": "Hate Speech"},
        {"text": "Bhadwa kutta nikal yahan se.", "label": "Hate Speech"},
        {"text": "Saale mar jaake kahin.", "label": "Hate Speech"},
        {"text": "Ek number ka mc hai tu.", "label": "Offensive"},
        {"text": "Teri maa ki ankh mc.", "label": "Hate Speech"},
        
        # Normal Text (Broad Vocabulary to fix TF-IDF imbalances)
        {"text": "You are amazing and I admire your work.", "label": "Normal"},
        {"text": "This movie was absolutely fantastic.", "label": "Normal"},
        {"text": "how are you", "label": "Normal"},
        {"text": "how are you today", "label": "Normal"},
        {"text": "what is your name", "label": "Normal"},
        {"text": "I am doing well, thank you.", "label": "Normal"},
        {"text": "hello there", "label": "Normal"},
        {"text": "good morning", "label": "Normal"},
        {"text": "good night", "label": "Normal"},
        {"text": "have a nice day", "label": "Normal"},
        {"text": "this is a test", "label": "Normal"},
        {"text": "where are you from", "label": "Normal"},
        {"text": "I really like this application.", "label": "Normal"},
        {"text": "the weather is beautiful today", "label": "Normal"},
        {"text": "can you help me with this problem", "label": "Normal"},
        {"text": "what time is it", "label": "Normal"},
        {"text": "I went to the store to buy some groceries.", "label": "Normal"},
        {"text": "my favorite food is pizza", "label": "Normal"},
        {"text": "I love listening to music.", "label": "Normal"},
        {"text": "please be careful", "label": "Normal"},
        {"text": "thank you so much", "label": "Normal"},
        {"text": "you are very welcome", "label": "Normal"},
        {"text": "I don't understand what you mean", "label": "Normal"},
        {"text": "let's go to the park", "label": "Normal"},
        {"text": "can we meet tomorrow", "label": "Normal"},
        {"text": "I am working on a college project.", "label": "Normal"},
        {"text": "this is the final presentation", "label": "Normal"},
        {"text": "Priyanshu Solanki designed this app.", "label": "Normal"},
        {"text": "Deviputra is a great name.", "label": "Normal"},
        
        # Normal Text (Hindi/Hinglish Vocabulary)
        {"text": "Bhai tu bohot accha insaan hai.", "label": "Normal"},
        {"text": "Aaj mausam bahut badhiya hai yaar.", "label": "Normal"},
        {"text": "Aap kaise ho bhaisaab, sab badhiya?", "label": "Normal"},
        {"text": "Mujhe India se bohot pyar hai.", "label": "Normal"},
        {"text": "Aap sahi ho bilkul.", "label": "Normal"},
        {"text": "Haan bhai ekdum sahi baat hai.", "label": "Normal"},
        {"text": "Main theek hoon, aap batao.", "label": "Normal"},
        {"text": "Sab kuch bohot accha hai.", "label": "Normal"},
        {"text": "Namaste, aapne bilkul sahi kaha.", "label": "Normal"},
        {"text": "Hello kaise ho sab badiya?", "label": "Normal"},
        {"text": "Wow this is very beautiful.", "label": "Normal"},
        {"text": "app sahi ho", "label": "Normal"},
        {"text": "aap sahi ho", "label": "Normal"},
        {"text": "kya haal hai bhai?", "label": "Normal"},
        {"text": "mera naam priyanshu hai", "label": "Normal"},
        {"text": "main college jaa raha hoon", "label": "Normal"},
        {"text": "aaj raat ko milte hain", "label": "Normal"},
        {"text": "khana kha liya kya?", "label": "Normal"},
        {"text": "bohot din baad mile yar", "label": "Normal"},
        
        # English Explicit Words For Vectorizer Learning
        {"text": "What the fuck are you doing?", "label": "Hate Speech"},
        {"text": "Fuck you idiot.", "label": "Hate Speech"},
        {"text": "This is absolute shit.", "label": "Offensive"},
        {"text": "Shut the fck up.", "label": "Hate Speech"},
        {"text": "You are a piece of crap.", "label": "Offensive"},
        {"text": "Listen to me you fuking idiot.", "label": "Hate Speech"},

        # ============================================================
        # CONTEXT-DEPENDENT HINDI/HINGLISH WORDS
        # These words have DUAL meanings — normal in one context,
        # vulgar/offensive in another. The surrounding words teach
        # TF-IDF which context is which.
        # ============================================================

        # --- GHANTA (Bell / Time vs vulgar dismissal) ---
        # Normal: Temple / Religious / School / Time context
        {"text": "Mandir mein ghanta baj raha hai.", "label": "Normal"},
        {"text": "Temple ka ghanta bahut sundar awaaz deta hai.", "label": "Normal"},
        {"text": "Pooja ke waqt ghanta bajana chahiye.", "label": "Normal"},
        {"text": "Aarti mein ghanta bajaya gaya.", "label": "Normal"},
        {"text": "School ka ghanta baj gaya ab chutti hai.", "label": "Normal"},
        {"text": "Ghanta bajao aarti shuru ho rahi hai.", "label": "Normal"},
        {"text": "Mandir mein subah subah ghanta bajta hai.", "label": "Normal"},
        {"text": "Church mein bhi ghanta bajate hain.", "label": "Normal"},
        {"text": "Mujhe aane mein ek ghanta lagega.", "label": "Normal"},
        {"text": "Do ghante baad class shuru hogi.", "label": "Normal"},
        {"text": "Kitne ghante lagenge wahan pahunchne mein?", "label": "Normal"},
        {"text": "Bas aadha ghanta bacha hai exam mein.", "label": "Normal"},
        {"text": "मंदिर में घंटा बज रहा है।", "label": "Normal"},
        {"text": "पूजा के समय घंटा बजाना शुभ होता है।", "label": "Normal"},
        {"text": "एक घंटा का समय लगेगा।", "label": "Normal"},
        {"text": "दो घंटे बाद मिलते हैं।", "label": "Normal"},
        # Offensive: Vulgar dismissal / slang
        {"text": "Ghanta farak padta hai tujhe.", "label": "Offensive"},
        {"text": "Ghanta kuch nahi hoga tera.", "label": "Offensive"},
        {"text": "Ghanta samajhta hai tu.", "label": "Offensive"},
        {"text": "Tujhe ghanta pata hai kuch.", "label": "Offensive"},
        {"text": "Ghanta kaam karta hai ye.", "label": "Offensive"},
        {"text": "Usko ghanta matlab hai tere se.", "label": "Offensive"},
        {"text": "घंटा फर्क पड़ता है तुझे।", "label": "Offensive"},
        {"text": "घंटा कुछ नहीं आता तुझे।", "label": "Offensive"},
        {"text": "तुझे घंटा पता है क्या हुआ।", "label": "Offensive"},

        # --- KUTTA (Pet dog vs insult) ---
        # Normal: Pet / Animal context
        {"text": "Mera kutta bahut pyaara hai.", "label": "Normal"},
        {"text": "Kutta ghar ka rakhwala hota hai.", "label": "Normal"},
        {"text": "Aaj kutta doctor ke paas le jaana hai.", "label": "Normal"},
        {"text": "Desi kutta bahut samajhdaar hota hai.", "label": "Normal"},
        {"text": "Humare mohalle mein ek kutta rehta hai.", "label": "Normal"},
        {"text": "Bacche kutta ke saath khel rahe hain.", "label": "Normal"},
        {"text": "Street kutta ko khana de do yaar.", "label": "Normal"},
        {"text": "मेरा कुत्ता बहुत समझदार और प्यारा है।", "label": "Normal"},
        {"text": "कुत्ता वफादार जानवर होता है।", "label": "Normal"},
        # Offensive: Targeted insult at a person
        {"text": "Tu kutta hai saala.", "label": "Hate Speech"},
        {"text": "Kutta hai tu ek number ka.", "label": "Hate Speech"},
        {"text": "Kutte ki tarah bhonkta hai tu.", "label": "Hate Speech"},
        {"text": "Kutta kamine nikal yahan se.", "label": "Hate Speech"},
        {"text": "कुत्ते कमीने तुझे बर्बाद कर दूंगा।", "label": "Hate Speech"},
        {"text": "तू कुत्ता है एक नंबर का।", "label": "Hate Speech"},

        # --- SAALA (Brother-in-law vs abuse) ---
        # Normal: Family relationship context
        {"text": "Mera saala aaj ghar aaya.", "label": "Normal"},
        {"text": "Saala bahut accha insaan hai.", "label": "Normal"},
        {"text": "Behen ka saala matlab wife ka bhai.", "label": "Normal"},
        {"text": "Mere saale ne shaadi mein bahut help ki.", "label": "Normal"},
        {"text": "Saala mere ghar mein rehta hai abhi.", "label": "Normal"},
        {"text": "Saale sahab se milke accha laga.", "label": "Normal"},
        {"text": "मेरा साला बहुत नेक इंसान है।", "label": "Normal"},
        {"text": "साले साहब हमारे घर पधारे हैं।", "label": "Normal"},
        # Offensive: Used as abuse
        {"text": "Saale tu mar ja.", "label": "Hate Speech"},
        {"text": "Saale tujhe choddunga nahi.", "label": "Hate Speech"},
        {"text": "Saala kamina hai tu.", "label": "Hate Speech"},
        {"text": "Saale ko sabak sikhao.", "label": "Offensive"},
        {"text": "साले तू जिंदा नहीं बचेगा।", "label": "Hate Speech"},
        {"text": "साला हरामखोर इंसान।", "label": "Hate Speech"},

        # --- TATTI (Baby mess / dirt vs vulgar insult) ---
        # Normal: Parenting / cleaning context
        {"text": "Bacche ki tatti saaf karo.", "label": "Normal"},
        {"text": "Bathroom mein tatti kar ke aao.", "label": "Normal"},
        {"text": "Kutta ne tatti kar di road pe.", "label": "Normal"},
        # Offensive: Used as insult
        {"text": "Tu tatti hai bilkul.", "label": "Offensive"},
        {"text": "Tatti jaisi shakal hai teri.", "label": "Hate Speech"},
        {"text": "Ye kaam tatti hai ekdum.", "label": "Offensive"},

        # --- HARAMI (Casual movie usage vs targeted insult) ---
        # Normal: Casual / movie / light context
        {"text": "Ye harami type ka movie character hai.", "label": "Normal"},
        {"text": "Harami ladka masti kar raha hai.", "label": "Normal"},
        # Offensive: Targeted at a person
        {"text": "Tu harami hai saale.", "label": "Hate Speech"},
        {"text": "Harami insaan hai tu puri duniya mein.", "label": "Hate Speech"},
        {"text": "Haramkhor saala nikal yahan se.", "label": "Hate Speech"},

        # --- KAMINA (Playful vs abusive) ---
        # Normal: Friendly playful banter
        {"text": "Arre kamina kitne din baad aaya tu.", "label": "Normal"},
        {"text": "Kamina hai tu but I love you bro.", "label": "Normal"},
        # Offensive: Targeted insult
        {"text": "Kamine tujhe zinda nahi chodunga.", "label": "Hate Speech"},
        {"text": "Kamina hai tu ek number ka.", "label": "Hate Speech"},
        {"text": "Kamine ko sabak sikhana padega.", "label": "Offensive"},

        # --- GADHA (Donkey animal vs stupid insult) ---
        # Normal: Animal context
        {"text": "Gaon mein gadha dekhne ko mila.", "label": "Normal"},
        {"text": "Gadha bahut mehnati jaanwar hota hai.", "label": "Normal"},
        {"text": "Baccho ne gadhe pe sawari ki.", "label": "Normal"},
        {"text": "गधा एक बहुत मेहनती जानवर है।", "label": "Normal"},
        # Offensive: Calling someone stupid
        {"text": "Tu gadha hai kya itna bhi nahi aata.", "label": "Offensive"},
        {"text": "Gadhe jaise kaam karta hai.", "label": "Offensive"},
        {"text": "Ek number ka gadha hai ye banda.", "label": "Offensive"},
        {"text": "एक नंबर का गधा इंसान है तू।", "label": "Offensive"},

        # --- ULLU (Owl bird vs fool insult) ---
        # Normal: Bird / nature context
        {"text": "Raat ko ullu dikhta hai jungle mein.", "label": "Normal"},
        {"text": "Ullu bahut intelligent bird hota hai.", "label": "Normal"},
        {"text": "Zoo mein ullu dekha aaj.", "label": "Normal"},
        {"text": "उल्लू रात में जागता है।", "label": "Normal"},
        # Offensive: Calling someone a fool
        {"text": "Tu ullu ka pattha hai.", "label": "Offensive"},
        {"text": "Ullu hai kya dimag nahi hai.", "label": "Offensive"},
        {"text": "Ullu ka pattha samajh nahi aata.", "label": "Offensive"},
        {"text": "उल्लू का पट्ठा है क्या तू।", "label": "Offensive"},

        # --- BANDAR (Monkey animal vs insult) ---
        # Normal: Animal / zoo context
        {"text": "Zoo mein bandar bahut masti karte hain.", "label": "Normal"},
        {"text": "Bandar ko kela bahut pasand hai.", "label": "Normal"},
        {"text": "Bacche bandar dekhne gaye the.", "label": "Normal"},
        # Offensive: Targeted insult
        {"text": "Bandar jaisi shakal hai teri.", "label": "Offensive"},
        {"text": "Tu bandar hai kya itna ulta seedha.", "label": "Offensive"}
    ]
    
    # Repeat the data heavily to have enough samples for an ML train/test split without losing features
    df = pd.DataFrame(data)
    df = pd.concat([df]*200, ignore_index=True)
    
    # Save directly to data folder
    filepath = os.path.join('data', 'sample.csv')
    df.to_csv(filepath, index=False)
    print(f"Created massively expanded sample dataset at {filepath}")

if __name__ == "__main__":
    create_sample_data()
