class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_token = False
        self.token_id = None

class TrieTokenizer:
    def __init__(self):
        self.root = TrieNode()
        self.id_to_token = {}
        self.token_to_id = {}
        self.next_id = 0

    def insert(self, token):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # Mark the end of a token
        node.end_of_token = True
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            node.token_id = self.next_id
            self.next_id += 1

    def fit(self, corpus_tokens):
        # Build trie from each token in the corpus
        for token in corpus_tokens:
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
                # No match found, treat single char as token
                # Insert single character so it has an ID
                char_token = text[i]
                if char_token not in self.token_to_id:
                    self.insert(char_token)
                tokens.append(self.token_to_id[char_token])
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids)

# Example usage
corpus = "The quick brown fox jumps over the lazy dog a b c d e f g h i j k l m n o p q r s t u v w x y z"
corpus_tokens = corpus.split()  # or any list of tokens

tokenizer = TrieTokenizer()
tokenizer.fit(corpus_tokens)

test_text = "The quick brown fox y z boab"
encoded = tokenizer.tokenize(test_text)
decoded = tokenizer.decode(encoded)

print("Encoded IDs:", encoded)
print("Decoded text:", decoded)