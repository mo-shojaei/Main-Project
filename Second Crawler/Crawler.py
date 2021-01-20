{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from pygments import highlight\n",
    "from pygments.lexers import JsonLexer\n",
    "from pygments.formatters import TerminalFormatter\n",
    "\n",
    "from google_play_scraper import Sort, reviews, app\n",
    "\n",
    "%matplotlib inline\n",
    "%config InlineBackend.figure_format='retina'\n",
    "\n",
    "sns.set(style='whitegrid', palette='muted', font_scale=1.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "app_packages = [\n",
    "  'com.anydo',\n",
    "  'com.todoist',\n",
    "  'com.ticktick.task',\n",
    "  'com.habitrpg.android.habitica',\n",
    "  'cc.forestapp',\n",
    "  'prox.lab.calclock',\n",
    "  'com.gmail.jmartindev.timetune',\n",
    "  'com.tasks.android',\n",
    "  'com.appgenix.bizcal'\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:05<00:00,  1.73it/s]\n"
     ]
    }
   ],
   "source": [
    "app_infos = []\n",
    "\n",
    "for ap in tqdm(app_packages):\n",
    "  info = app(ap, lang='en', country='us')\n",
    "  del info['comments']\n",
    "  app_infos.append(info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_json(json_object):\n",
    "  json_str = json.dumps(\n",
    "    json_object,\n",
    "    indent=2,\n",
    "    sort_keys=True,\n",
    "    default=str\n",
    "  )\n",
    "  print(highlight(json_str, JsonLexer(), TerminalFormatter()))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "app_infos_df = pd.DataFrame(app_infos)\n",
    "app_infos_df.to_csv('apps.csv', index=None, header=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████████████████████████████████████████████████| 9/9 [05:08<00:00, 35.22s/it]\n"
     ]
    }
   ],
   "source": [
    "app_reviews = []\n",
    "\n",
    "for ap in tqdm(app_packages):\n",
    "  for score in list(range(1, 6)):\n",
    "    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:\n",
    "      rvs, _ = reviews(\n",
    "        ap,\n",
    "        lang='en',\n",
    "        country='us',\n",
    "        sort=sort_order,\n",
    "        count= 200 if score == 5 else 1000,\n",
    "        filter_score_with=score\n",
    "      )\n",
    "      for r in rvs:\n",
    "        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'\n",
    "        r['appId'] = ap\n",
    "      app_reviews.extend(rvs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54832"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(app_reviews)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "app_reviews_df = pd.DataFrame(app_reviews)\n",
    "app_reviews_df.to_csv('reviews.csv', index=None, header=True)"
   ]
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
