{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'gw_utility'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-4-6cfea98edb62>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# outer_import_2.7.py\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0msys\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mgw_utility\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbook\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'gw_utility'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import nltk\n",
    "import inflect\n",
    "import re\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "\n",
    "def to_lowercase(words):\n",
    "    # Convert all characters to lowercase from list of tokenized words\n",
    "    new_words = []\n",
    "    for word in words:\n",
    "        new_word = word.lower()\n",
    "        new_words.append(new_word)\n",
    "    return new_words\n",
    "\n",
    "def remove_punctuation(words):\n",
    "    # Remove punctuation from list of tokenized words\n",
    "    new_words = []\n",
    "    for word in words:\n",
    "        new_word = re.sub(r'[^\\w\\s]', '', word)\n",
    "        if new_word != '':\n",
    "            new_words.append(new_word)\n",
    "    return new_words\n",
    "\n",
    "def remove_numbers(words):\n",
    "    # Replace all interger occurrences in list of tokenized words with textual representation\n",
    "    p = inflect.engine()\n",
    "    new_words = []\n",
    "    for word in words:\n",
    "        if word.isdigit():\n",
    "            new_word = re.sub('[0-9]+', '', word)\n",
    "            new_words.append(new_word)\n",
    "        else:\n",
    "            new_words.append(word)\n",
    "    return new_words\n",
    "\n",
    "def remove_stopwords(words):\n",
    "    # Remove stop words from list of tokenized words\n",
    "    new_words = []\n",
    "    for word in words:\n",
    "        if word not in stopwords.words('english'):\n",
    "            new_words.append(word)\n",
    "    return new_words\n",
    "\n",
    "def lemmatize_verbs(words):\n",
    "    # Lemmatize verbs in list of tokenized words\n",
    "    lemmatizer = WordNetLemmatizer()\n",
    "    lemmas = []\n",
    "    for word in words:\n",
    "        lemma = lemmatizer.lemmatize(word, pos='v')\n",
    "        lemmas.append(lemma)\n",
    "    return lemmas\n",
    "\n",
    "def normalize(words):\n",
    "    # Data Cleansing Pipeline\n",
    "    words = to_lowercase(words)\n",
    "    words = remove_punctuation(words)\n",
    "    words = remove_numbers(words)\n",
    "    words = remove_stopwords(words)\n",
    "    lemmas = lemmatize_verbs(words)\n",
    "    return lemmas\n",
    "\n",
    "\n",
    "\n",
    "word_list = []\n",
    "word_list_dict = {}\n",
    "for idx, row  in df.iterrows():\n",
    "    words = nltk.word_tokenize(row['content'])\n",
    "    words = normalize(words)\n",
    "    word_list.append(words)\n",
    "    word_list_dict[idx] = word_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
