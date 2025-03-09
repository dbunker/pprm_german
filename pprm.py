import pandas as pd
import math

# =======================
# Utility Functions
# =======================

def is_vowel(phone):
    """
    Determines if phone is considered a vowel in German CPSAMPA.
    """
    vowels = set(["a","A","e","E","i","I","o","O","u","U","y","Y","2","9","@","3","aI","aU","OY"])
    return phone[0] in vowels


def is_front_vowel(phone):
    """
    Check if phone is considered a front vowel. This is relevant for [ç].
    """
    # Front vowels in German: i, e, ɛ, y, ø, ...
    front_vowels = set(["i","I","e","E","y","Y","2","9","e:","i:","ø","œ","æ"])
    return any(phone.startswith(vf) for vf in front_vowels)


def is_back_vowel(phone):
    """
    Check if phone is considered a back vowel. This is relevant for [x].
    """
    # Back vowels in German: a, o, u, ɑ, ɒ, ...
    back_vowels = set(["a","A","o","O","u","U","o:","u:","ɔ","ʊ","ɑ"])
    return any(phone.startswith(vb) for vb in back_vowels)


def phone_sequence(cpsampa):
    """
    Splits the CPSAMPA transcription into a list of phones by '.'
        Phone: The distinct speech sound
        Phoneme: Abstract unit of sound that when changed can alter the meaning of a word (cat vs. bat)
        Allophone: A phone associated with a specific phoneme, the way the phoneme manefests
    """
    
    tokens = cpsampa.split('.')
    return [t for t in tokens if t]


def shannon_entropy(prob1, prob2):
    """
    Calculates binary Shannon entropy: - (p1 log2 p1 + p2 log2 p2), adjusting for p=0.
    """
    H = 0
    if prob1 > 0:
        H -= prob1 * math.log2(prob1)
    if prob2 > 0:
        H -= prob2 * math.log2(prob2)
    return H

# =======================
# Environment Classifiers for German
# =======================

def env_x_c(phones, i):
    """
    Classifies environments for [x] vs. [ç].
    1) WordInitial: 
        Phone at start of word
    2) WordFinal_AfterBackVowel: 
        Preceding is a back vowel
    3) WordFinal_AfterFrontOrCons:
        Otherwise
    4) Medial_AfterBackVowel:
        Not final and preceding is back vowel
    5) Medial_AfterFrontOrCons:
        Not final and otherwise
    """
    if i == 0:
        return "WordInitial"
    
    # Check the previous phone
    prev_phone = phones[i-1]
    
    # Check if is last in word
    if i == len(phones) - 1:
        # Check for back vowel
        if is_back_vowel(prev_phone):
            return "WordFinal_AfterBackVowel"
        else:
            return "WordFinal_AfterFrontOrCons"
    
    # Otherwise medial
    if is_back_vowel(prev_phone):
        return "Medial_AfterBackVowel"
    else:
        return "Medial_AfterFrontOrCons"


def env_s_S(phones, i):
    """
    Classifies environments for [s] vs. [ʃ].
    1) WordInitial_s_plus_stop:
        Check if next phone is [p] or [t] for typical [ʃp], [ʃt] clusters
    2) WordInitial_s_alone:
        Simple 's'
    3) WordFinal:
        End of word, could also check if before a vowel vs. before a consonant
    4) WordMedial:
        - Otherwise
    """
    if i == 0:
        # Check if next phone is 'p' or 't'
        if len(phones) > 1:
            if phones[i+1] in ['p', 't']:
                return "WordInitial_s_plus_stop"
        return "WordInitial_s_alone"
    
    if i == len(phones) - 1:
        return "WordFinal"
    
    # Otherwise medial
    return "WordMedial"


def env_d_t(phones, i):
    """
    Classifies environments for [d] vs. [t] in German.
      1) WordInitial:
        At beginning of word
      2) WordFinal:
        At end of word, may want to check for final devoicing
      3) Intervocalic (V_V):
        In between vowels
      4) ClusterOrOther:
        Cluster or other
    """
    if i == 0:
        return "WordInitial"
    
    if i == len(phones) - 1:
        return "WordFinal"
    
    # Check neighbors
    left = phones[i-1]
    right = phones[i+1]
    
    if is_vowel(left) and is_vowel(right):
        return "Intervocalic_V_V"
    
    return "ClusterOrOther"

# ============================
# Probabilistic Phonological Relationship Model (PPRM) Algorithm
# ============================

def pprm_for_pair(df, phoneX, phoneY, env_function):
    """
    Collects type and token counts and calculates entropies for PPRM algorithm
    """
    counts_type = {}
    counts_token = {}
    counts_type_tracker = {}
    
    def ensure_env(env):
        if env not in counts_type:
            counts_type[env] = {phoneX: 0, phoneY: 0}
            counts_token[env] = {phoneX: 0, phoneY: 0}
            counts_type_tracker[env] = {phoneX: set(), phoneY: set()}
    
    for idx, row in df.iterrows():
        word_str = row["Word"]
        phono_str = row["Phono"]
        freq = row["Frequency"]
        phones = phone_sequence(phono_str)
        
        for i, ph in enumerate(phones):
            if ph == phoneX or ph == phoneY:
                env_label = env_function(phones, i)
                ensure_env(env_label)
                
                # Type frequency
                if word_str not in counts_type_tracker[env_label][ph]:
                    counts_type[env_label][ph] += 1
                    counts_type_tracker[env_label][ph].add(word_str)
                
                # Token frequency
                counts_token[env_label][ph] += freq
    
    # Compute entropies
    total_types = 0
    for env in counts_type:
        total_types += (counts_type[env][phoneX] + counts_type[env][phoneY])
    
    total_tokens = 0
    for env in counts_token:
        total_tokens += (counts_token[env][phoneX] + counts_token[env][phoneY])
    
    env_entropies_type = {}
    env_entropies_token = {}
    env_probs_type = {}
    env_probs_token = {}
    
    # Determine entropies for each environment
    for env in counts_type:
        X_type = counts_type[env][phoneX]
        Y_type = counts_type[env][phoneY]
        if (X_type + Y_type) > 0:
            pX_type = X_type / (X_type + Y_type)
            pY_type = 1 - pX_type
            H_type = shannon_entropy(pX_type, pY_type)
            env_entropies_type[env] = H_type
            env_probs_type[env] = (X_type + Y_type) / total_types
        
        X_token = counts_token[env][phoneX]
        Y_token = counts_token[env][phoneY]
        if (X_token + Y_token) > 0:
            pX_token = X_token / (X_token + Y_token)
            pY_token = 1 - pX_token
            H_token = shannon_entropy(pX_token, pY_token)
            env_entropies_token[env] = H_token
            env_probs_token[env] = (X_token + Y_token) / total_tokens
    
    overall_entropy_type = 0
    for env in env_entropies_type:
        overall_entropy_type += env_probs_type[env] * env_entropies_type[env]

    overall_entropy_token = 0
    for env in env_entropies_token:
        overall_entropy_token += env_probs_token[env] * env_entropies_token[env]

    result = {
        "pair": f"[{phoneX}]~[{phoneY}]",
        "type_entropy": overall_entropy_type,
        "token_entropy": overall_entropy_token,
        "env_type_counts": counts_type,
        "env_token_counts": counts_token,
        "env_type_entropies": env_entropies_type,
        "env_token_entropies": env_entropies_token,
        "env_type_probs": env_probs_type,
        "env_token_probs": env_probs_token
    }
    return result


def main():
    # German dataset relevant columns:
    #   Word: The word 
    #   Phono: The CPSAMPA string with phones separated by '.'
    #   Frequency: The token frequency in your corpus
    df = pd.read_csv("german.csv")
    
    # Pairs to analyze:
    pairs = [
        # (phoneX, phoneY, environment_classifier)
        ('x', 'C', env_x_c),    # [x] vs [ç]    'C' for [ç]
        ('s', 'S', env_s_S),    # [s] vs [ʃ]    'S' for [ʃ]
        ('d', 't', env_d_t),    # [d] vs [t]
    ]

    for (x, y, env_f) in pairs:
        res = pprm_for_pair(df, x, y, env_f)
        print("===========================================")
        print(f"Results for pair: {res['pair']}")
        print(f"  Type-based Entropy  = {res['type_entropy']:.4f}")
        print(f"  Token-based Entropy = {res['token_entropy']:.4f}")
        
        # Entropy and progabilites for environments
        print("  Environment (Type) Entropies:")
        for e, h_e in res["env_type_entropies"].items():
            print(f"    {e}: H(e)={h_e:.3f}, p(e)={res['env_type_probs'][e]:.4f}")
        
        print("  Environment (Token) Entropies:")
        for e, h_e in res["env_token_entropies"].items():
            print(f"    {e}: H(e)={h_e:.3f}, p(e)={res['env_token_probs'][e]:.4f}")


if __name__ == "__main__":
    main()
