from collections import Counter

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_token = False
        self.token_id = None

class TrieTokenizer:
    def __init__(self, max_tokens=50):
        self.root = TrieNode()
        self.id_to_token = {}
        self.token_to_id = {}
        self.next_id = 0
        self.max_tokens = max_tokens

    def insert(self, token):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_token = True
        # Only assign a new ID if this exact token isn't already in self.token_to_id
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            node.token_id = self.next_id
            self.next_id += 1

    def fit(self, corpus_tokens):
        """
        1. Collect all single characters from the corpus and insert them (ensuring coverage).
        2. Use remaining slots for the most frequent multi-character tokens.
        """
        freqs = Counter(corpus_tokens)
        unique_chars = set()
        for token in corpus_tokens:
            for ch in token:
                unique_chars.add(ch)

        # Step 1: Insert all unique single characters (e.g. 'a', 'b') to guarantee coverage
        for ch in sorted(unique_chars):
            if self.next_id < self.max_tokens:
                self.insert(ch)

        # Step 2: Insert the most common multi-character tokens until we run out of space
        # Filter out single characters from the counter, so we only consider multi-char tokens
        multi_char_tokens = [(t, f) for t, f in freqs.items() if len(t) > 1]
        multi_char_tokens.sort(key=lambda x: x[1], reverse=True)

        for token, _ in multi_char_tokens:
            if self.next_id >= self.max_tokens:
                break
            self.insert(token)

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            node = self.root
            last_match_id = None
            last_match_pos = i
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.end_of_token:
                    last_match_id = node.token_id
                    last_match_pos = j
            if last_match_id is not None:
                tokens.append(last_match_id)
                i = last_match_pos
            else:
                # If there's no match, fallback to single-character coverage
                char_token = text[i]
                # Because we guaranteed single-char coverage in fit(), it must exist
                tokens.append(self.token_to_id[char_token])
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids)

# Example usage:
if __name__ == "__main__":
    tokenizer = TrieTokenizer(max_tokens=5)
    # "aa", "a", "b" – let's say 'a' and 'b' appear as single chars, "aa" is multi-char
    corpus_tokens = ["aa", "aa", "a", "b", "b", "b"]
    tokenizer.fit(corpus_tokens)

    # Let's see what ended up in the vocabulary
    print("ID to token:", tokenizer.id_to_token)
    print("Token to ID:", tokenizer.token_to_id)

    # Now tokenize this list
    test_text = "aab"
    token_ids = tokenizer.tokenize(test_text)
    print("Token IDs:", token_ids)
    print("Decoded:", tokenizer.decode(token_ids))