import os
import re
from collections import Counter

def main():
    data_dir = '/data/2022-DIT065-DAT470/gutenberg/'
    sample_file = data_dir + '060/06049.txt'
    regex = re.compile('[^a-zA-Z ]')
    word_counter = Counter()
    with open(sample_file, 'r') as f:
        for line in f:
            # Remove all non-aphabetical characters
            cleaned_line = regex.sub('', line.lower())
            words_in_line = cleaned_line.split(' ')
            word_counter.update(words_in_line)
    num_unique_words = len(list(word_counter))
    print(f'num_words\t{word_counter.total()}')
    print(f'num_unique_words\t{num_unique_words}')
    words = list(word_counter.elements())
    avg_len = sum(map(len, words)) / len(words)
    print(f'avg_len\t{avg_len}')

    sorted_words = sorted(words)
    common_prefixes = set()
    for i, w1 in enumerate(sorted_words):
        for w2 in sorted_words[i+1:]:
            if w1 != '' and w2 != '':
                if w1[0] != w2[0]:
                    # Since they are sorted, skip searching the remaining part
                    break
            common_prefixes.add(os.path.commonprefix([w1, w2]))
    print(common_prefixes)

if __name__ == '__main__':
    main()